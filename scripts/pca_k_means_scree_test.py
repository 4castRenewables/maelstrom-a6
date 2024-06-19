"""
apptainer run --cleanenv --env OPENBLAS_NUM_THREADS=1 -B /p/home/jusers/$USER/juwels/code/a6:/opt/a6 /p/project/deepacf/$USER/a6-cuda.sif python /opt/a6/scripts/pca_k_means_scree_test.py  # noqa: E501
"""
import concurrent.futures
import contextlib
import logging
import os
import pathlib
import time
from collections.abc import Callable

import joblib
import mantik.mlflow
import numpy as np
import scipy.sparse
import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.utils.sparsefuncs
import xarray as xr

import a6

logger = logging.getLogger(__name__)


N_CLUSTERS = int(os.getenv("N_CLUSTERS_KMEANS", 40))
Ks = list(range(1, N_CLUSTERS + 1))
ds = xr.open_dataset(
    pathlib.Path(
        "/p/project/deepacf/emmerich1/data/ecmwf_era5/era5_pl_1964_2023_12.nc"
    )
)
data_dir = pathlib.Path("/p/project/deepacf/emmerich1/data")

pca_dir = data_dir / "pca"
pca_dir.mkdir(exist_ok=True, parents=True)

kmeans_dir = data_dir / "kmeans"
kmeans_dir.mkdir(exist_ok=True, parents=True)


@contextlib.contextmanager
def measure_time(message: str = ""):
    if message:
        logger.info(message)

    start = time.time()

    yield

    duration = time.time() - start
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    duration_str = f"{minutes:02d}:{seconds:02d} (MM:SS)"
    if message:
        logger.info("%s finished in %s", message, duration_str)
    else:
        logger.info("Finished in %s", duration_str)


def read_from_disk_if_exists(
    path: pathlib.Path,
    method,
    fit: bool = False,
    data: np.ndarray | None = None,
    **kwargs,
):
    if path.exists():
        with measure_time(f"Loading {method} from disk at {path.as_posix()}"):
            return joblib.load(path)
    else:
        path.parent.mkdir(exist_ok=True, parents=True)
        with measure_time(f"Applying {method} with {kwargs}"):
            result = method(**kwargs)

            if fit:
                if data is None:
                    raise ValueError("Fitting requires 'data' argument")

                with measure_time(f"Fitting {method} to data"):
                    result = result.fit(data)

            joblib.dump(result, path)
            return result


def calculate_ssd(k: int, data: np.ndarray, type: str, n_pcs: int) -> float:
    kmeans_path = kmeans_dir / f"kmeans_{type}_n_pcs_{n_pcs}_k_{k}.joblib"

    with measure_time(f"Fitting k-Means for {type.upper()} ({n_pcs=}, {k=})"):
        kmeans = read_from_disk_if_exists(
            path=kmeans_path,
            method=sklearn.cluster.KMeans,
            n_clusters=k,
            fit=True,
            data=data,
        )

        return kmeans.inertia_


def transform_into_pc_space_and_standardize(
    pca: sklearn.decomposition.PCA, X: np.ndarray, n_components: int
):
    X = pca._validate_data(
        X,
        accept_sparse=("csr", "csc"),
        dtype=[np.float64, np.float32],
        reset=False,
    )
    if pca.mean_ is not None:
        if scipy.sparse.issparse(X):
            X = sklearn.utils.sparsefuncs._implicit_column_offset(X, pca.mean_)
        else:
            X = X - pca.mean_
    X_transformed = X @ pca.components_[:n_components, :].T
    if pca.whiten:
        X_transformed /= np.sqrt(pca.explained_variance_[:n_components])
    X_transformed = sklearn.preprocessing.StandardScaler().fit_transform(
        X_transformed
    )
    return X_transformed


def transform_data(
    kpca: sklearn.decomposition.KernelPCA,
    X: np.ndarray,
) -> np.ndarray:
    logger.info(
        "Transforming data into kernel PC space (n_pcs=%i)",
        kpca.eigenvalues_.shape[0],
    )
    transformed = kpca.transform(X)
    transformed = sklearn.preprocessing.StandardScaler().fit_transform(
        transformed
    )
    return transformed


def calculate_ssd_pca(
    pca: sklearn.decomposition.PCA, n_pcs: int, data: np.ndarray
):
    pca_tansformed_path = pca_dir / f"pca_{n_pcs}_pcs_transformed.joblib"

    with measure_time(f"Transforming PCA result ({n_pcs=})"):
        transformed = read_from_disk_if_exists(
            path=pca_tansformed_path,
            method=transform_into_pc_space_and_standardize,
            pca=pca,
            X=data,
            n_components=n_pcs,
        )

    with measure_time(f"Calculating SSDs for PCA ({n_pcs=})"):
        ssds = a6.utils.parallelize.parallelize_with_futures(
            calculate_ssd,
            kwargs=[
                dict(k=k, data=transformed, type="pca", n_pcs=n_pcs) for k in Ks
            ],
            executor_type=concurrent.futures.ThreadPoolExecutor,
        )

    return {k: ssd for k, ssd in zip(Ks, ssds, strict=True)}


def calculate_ssd_kpca(n_pcs: int, data: np.ndarray):
    kpca_path = pca_dir / f"kpca_{n_pcs}_pcs.joblib"
    kpca_tansformed_path = pca_dir / f"kpca_{n_pcs}_pcs_transformed.joblib"

    logger.info("Calculating SSDs for kPCA (n_pcs=%i)", n_pcs)

    if kpca_tansformed_path.exists():
        logger.info("Reading kPCA-transformed data from disk")
        transformed = joblib.load(kpca_tansformed_path)
    else:
        kpca = sklearn.decomposition.KernelPCA(n_components=n_pcs)

        with measure_time(f"Fitting kPCA ({n_pcs=}) and transforming data"):
            transformed = kpca.fit_transform(data)
            joblib.dump(kpca, kpca_path)
            del kpca
    
        with measure_time("Normalizing data"):
            transformed = sklearn.preprocessing.StandardScaler().fit_transform(transformed)
            joblib.dump(transformed, kpca_tansformed_path)

    with measure_time(f"Calculating SSDs for kPCA {n_pcs=}"):
        ssds = [
            calculate_ssd(k=k, data=transformed, type="kpca", n_pcs=n_pcs)
            for k in Ks
        ]
    return {k: ssd for k, ssd in zip(Ks, ssds, strict=True)}


def calculate_ssds(
    method: Callable, method_name: str, n_pcs: int, data: np.ndarray, n_pcs_start: int = 1, parallelize: bool = True, **kwargs
) -> dict:
    n_pcs_range = range(n_pcs_start, n_pcs + 1)

    if parallelize:
        with measure_time(f"Calculating SSDs for {method_name}"):
            ssds = a6.utils.parallelize.parallelize_with_futures(
                method,
                kwargs=[
                    dict(n_pcs=n_pcs, data=data, **kwargs) for n_pcs in n_pcs_range
                ],
                executor_type=concurrent.futures.ProcessPoolExecutor,
            )
    else:
        ssds = [
            method(n_pcs=n_pcs, data=data, **kwargs)
            for n_pcs in n_pcs_range
        ]

    return {pcs: ssd for pcs, ssd in zip(n_pcs_range, ssds, strict=True)}


if __name__ == "__main__":
    a6.utils.logging.create_logger(
        global_rank=0,
        local_rank=0,
        verbose=False,
    )

    # Below PCs cover 80% of the total variance.
    n_pcs_kpca = int(os.getenv("N_PCS_KPCA", 32))
    n_pcs_pca = int(os.getenv("N_PCS_PCA", 80))
    n_pcs_kpca_start = int(os.getenv("N_PCS_KPCA_START", 1))
    n_pcs_pca_start = int(os.getenv("N_PCS_PCA_START", 1))

    data_path = os.getenv(
        "PREPROCESSED_DATA_PATH",
        "/p/project/deepacf/emmerich1/data/ecmwf_era5/era5_pl_1964_2023_12_preprocessed_for_pca.nc",  # noqa: E501
    )

    with measure_time("Reading data"):
        data = xr.open_dataset(data_path).to_dataarray().values[0]

    if "RUN_KPCA" not in os.environ:
        # For PCA, transformation can always be done with full PCA.
        # Thus, we load precomputed PCA with 500 components from disk.
        n_pcs_pca_full = 500
        pca_path = pca_dir / f"pca_{n_pcs_pca_full}_pcs.joblib"
        pca = read_from_disk_if_exists(
            path=pca_path,
            method=sklearn.decomposition.PCA,
            n_components=n_pcs_pca_full,
            fit=True,
            data=data,
        )
        ssds_pca = calculate_ssds(
            method=calculate_ssd_pca,
            method_name="PCA",
            n_pcs=n_pcs_pca,
            n_pcs_start=n_pcs_pca_start,
            pca=pca,
            data=data,
        )

        joblib.dump(
            ssds_pca,
            data_dir / "scree-test-results-pca.dict",
        )

        for k in Ks:
            mantik.mlflow.log_metrics(
                {
                    f"ssd_pca_{pcs}": ssds_per_k[k]
                    for pcs, ssds_per_k in ssds_pca.items()
                },
                step=k,
            )
    else:
        ssds_kpca = calculate_ssds(
            method=calculate_ssd_kpca,
            method_name="kPCA",
            n_pcs=n_pcs_kpca,
            n_pcs_start=n_pcs_kpca_start,
            data=data,
            # Parallelizing while using the entire dataset exceeds 512GB memory.
            parallelize=False,
        )

        joblib.dump(
            ssds_kpca,
            data_dir / "scree-test-results-kpca.dict",
        )

        for k in Ks:
            mantik.mlflow.log_metrics(
                {
                    f"ssd_kpca_{pcs}": ssds_per_k[k]
                    for pcs, ssds_per_k in ssds_kpca.items()
                },
                step=k,
            )
            time.sleep(0.1)
