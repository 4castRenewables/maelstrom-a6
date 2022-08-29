import functools
import json
import pathlib
import typing as t

import boto3
import lifetimes
import pandas as pd
import sklearn.decomposition as decomposition


def read_data(
    path: t.Union[str, pathlib.Path],
    n_components: int,
    use_varimax: bool,
) -> pd.DataFrame:
    """Run PCA on ECMWF IFS HRES data."""
    ds = lifetimes.datasets.EcmwfIfsHres(
        paths=[path],
        overlapping=False,
    )
    data = ds.as_xarray()["t"]

    modes = [lifetimes.modes.Modes(feature=data)]

    pca_partial_method = functools.partial(
        lifetimes.modes.methods.spatio_temporal_pca,
        algorithm=decomposition.PCA(n_components=n_components),
        time_coordinate="time",
        latitude_coordinate="latitude",
    )
    [pca] = lifetimes.modes.determine_modes(
        modes=modes, method=pca_partial_method
    )

    if use_varimax:
        data = pca.transform_with_varimax_rotation()
    else:
        data = pca.transform()
    return pd.DataFrame(data)


if __name__ == "__main__":

    parser = lifetimes.cli.aws.create_sagemaker_inference_parser()
    parser = lifetimes.cli.inference.create_parser(parser)
    args = parser.parse_args()

    df = read_data(
        path=args.data,
        n_components=args.n_components,
        use_varimax=args.use_varimax,
    )

    runtime = boto3.client("runtime.sagemaker")
    payload = df.to_json(orient="split")

    runtime_response = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(runtime_response["Body"].read().decode())

    print(f"Payload: {payload}")
    print(f"Prediction: {result}")
