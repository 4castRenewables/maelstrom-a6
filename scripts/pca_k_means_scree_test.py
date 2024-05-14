"""
apptainer run --cleanenv --env OPENBLAS_NUM_THREADS=1 -B /p/home/jusers/$USER/juwels/code/a6:/opt/a6 /p/project/deepacf/$USER/a6-cuda.sif python /opt/a6/scripts/pca_k_means_scree_test.py
"""
import pathlib
import time
import contextlib
import joblib

import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing
import matplotlib.pyplot as plt
import scipy.sparse
import sklearn.utils.sparsefuncs
import xarray as xr
import mantik.mlflow

import a6


N_CLUSTERS = 40
Ks = list(range(1, N_CLUSTERS + 1))
ds = xr.open_dataset(pathlib.Path(
    "/p/project/deepacf/emmerich1/data/ecmwf_era5/era5_pl_1964_2023_12.nc"
))
data_dir = pathlib.Path("/p/project/deepacf/emmerich1/data")

pca_dir = data_dir / "pca"
pca_dir.mkdir(exist_ok=True, parents=True)

kmeans_dir = data_dir / "kmeans"
kmeans_dir.mkdir(exist_ok=True, parents=True)

@contextlib.contextmanager
def measure_time(message: str = ""):
    if message:
        print(message)
        
    start = time.time()
    
    yield
    
    duration = time.time() - start
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    duration_str = f"{minutes:02d}:{seconds:02d} (MM:SS)"
    if message:
        print(f"{message} finished in {duration_str}")
    else:
        print(f"Finished in {duration_str}")

def calculate_ssd(k: int, data, type: str, n_pcs: int) -> float:
    km = sklearn.cluster.KMeans(n_clusters=k)
    km.fit(data)
    
    joblib.dump(km, kmeans_dir / f"kmeans_{type}_n_pcs_{n_pcs}_{k=}.joblib")
    
    return km.inertia_


def transform_into_pc_space_and_standardize(pca, X, n_components: int):
    X = pca._validate_data(
        X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
    )
    if pca.mean_ is not None:
        if scipy.sparse.issparse(X):
            X = sklearn.utils.sparsefuncs._implicit_column_offset(X, pca.mean_)
        else:
            X = X - pca.mean_
    X_transformed = X @ pca.components_[:n_components, :].T
    if pca.whiten:
        X_transformed /= xp.sqrt(pca.explained_variance_[:n_components])
    X_transformed = sklearn.preprocessing.StandardScaler().fit_transform(X_transformed)
    return X_transformed

with measure_time("Reading data"):
    data = (
        xr.open_dataset(
            "/p/project/deepacf/emmerich1/data/ecmwf_era5/era5_pl_1964_2023_12_preprocssed_for_pca.nc"
        )
        .to_dataarray()
        .values[0]
    )


def read_from_disk_if_exists(path: pathlib.Path, method, **kwargs):
    if path.exists():
        with measure_time(f"Loading {method} from disk at {path.as_posix()}"):
            return joblib.load(path)
    else:
        path.parent.mkdir(exist_ok=True, parents=True)
        with measure_time(f"Applying {method}"):
            result = method(**kwargs)
            joblib.dump(result, path)
            return result


def transform_data(
    pca,
    data,
):
    transformed = pca.fit_transform(data)
    transformed = sklearn.preprocessing.StandardScaler().fit_transform(transformed)
    return transformed


def calculate_ssd_pca(pca, n_pcs: int):
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
            kwargs=[dict(k=k, data=transformed, type="pca", n_pcs=n_pcs) for k in Ks],
        )

    return {k: ssd for k, ssd in zip(Ks, ssds, strict=True)}
    
def calculate_ssd_kpca(n_pcs: int):
    kpca_path = pca_dir / f"kpca_{n_pcs}_pcs.joblib"
    kpca_tansformed_path = pca_dir / f"kpca_{n_pcs}_pcs_transformed.joblib"

    kpca = read_from_disk_if_exists(
        path=kpca_path,
        method=sklearn.decomposition.KernelPCA,
        n_components=n_pcs,
    )
    
    with measure_time(f"Transforming kPCA result ({n_pcs=})"):
        transformed = read_from_disk_if_exists(
            path=kpca_tansformed_path,
            method=transform_data,
            pca=kpca,
            data=data,
        )

    with measure_time(f"Calculating SSDs for kPCA {n_pcs=}"):
        ssds = a6.utils.parallelize.parallelize_with_futures(
            calculate_ssd,
            kwargs=[dict(k=k, data=transformed, type="kpca", n_pcs=n_pcs) for k in Ks],
        )

    mantik.mlflow.log_metrics(
        {
            f"ssd_pca_k_{k}": ssd
            for k, ssd in zip(Ks, ssds, strict=True)
        },
        step=n_pcs,
    )

    return {k: ssd for k, ssd in zip(Ks, ssds, strict=True)}

def calculate_ssds(method, method_name: str, n_pcs: int, **kwargs) -> dict:
    with measure_time(f"Calculating SSDs for {method_name}"):
        n_pcs_range = range(1, n_pcs + 1)
        ssds = a6.utils.parallelize.parallelize_with_futures(
            method,
            kwargs=[dict(n_pcs=n_pcs, **kwargs) for n_pcs in n_pcs_range],
        )
        result = {pcs: ssd for pcs, ssd in zip(n_pcs_range, ssds, strict=True)}

    for k in Ks:
        mantik.mlflow.log_metrics(
            {
                f"ssd_{method_name.lower()}_{pcs}": ssds_per_k[k]
                for pcs, ssds_per_k in ssds.items()
            },
            step=k,
        )

    return result

if __name__ == "__main__":
    n_pcs_kpca = 15
    # For PCA, transformation can always be done with full PCA.
    # Thus, we load precomputed PCA with 80 components from disk.
    n_pcs_pca = 80
    pca_path = pca_dir / f"pca_{n_pcs_pca}_pcs.joblib"
    pca = read_from_disk_if_exists(
        path=pca_path,
        method=sklearn.decomposition.PCA,
        n_components=n_pcs_pca,
    )

    ssds_pca = calculate_ssds(
        method=calculate_ssd_pca,
        method_name="PCA",
        n_pcs=n_pcs_pca,
        pca=pca,
    )
    ssds_kpca = calculate_ssds(
        method=calculate_ssd_kpca,
        method_name="kPCA",
        n_pcs=n_pcs_kpca,
    )

    joblib.dump({"ssds_pca": ssds_pca, "ssds_kpca": ssds_kpca}, data_dir / "scree-test-results.dict")
