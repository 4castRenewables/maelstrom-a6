import argparse
import functools
import json
import pathlib
import typing as t

import boto3
import distutils.util
import lifetimes
import pandas as pd


def read_data(
    path: t.Union[str, pathlib.Path],
    variance_ratio: float,
    use_varimax: bool,
) -> pd.DataFrame:
    """Run PCA on ECMWF IFS HRES data."""
    ds = lifetimes.features.EcmwfIfsHresDataset(
        paths=[path],
        overlapping=False,
    )
    data = ds.as_xarray()["t"]

    modes = [lifetimes.modes.Modes(feature=data)]

    pca_partial_method = functools.partial(
        lifetimes.modes.methods.spatio_temporal_principal_component_analysis,
        variance_ratio=variance_ratio,
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

    def string_to_bool(s: str) -> bool:
        return bool(distutils.util.strtobool(s))

    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--variance-ratio", nargs="+", default=0.95, type=float)
    parser.add_argument(
        "--use-varimax", nargs="+", default=False, type=string_to_bool
    )
    args = parser.parse_args()

    df = read_data(
        path=args.data,
        variance_ratio=args.variance_ratio,
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
