import json
import pathlib

import a6.cli.arguments as arguments
import a6.cli.main as main
import a6.cli.options as _options
import a6.modes.methods as methods
import boto3
import click
import pandas as pd
import sklearn.decomposition as decomposition
import xarray as xr


@main.cli.command("inference")
@arguments.sagemaker.ENDPOINT
@_options.data.WEATHER_DATA
@_options.data.LEVEL
@_options.pca.N_COMPONENTS
@_options.pca.USE_VARIMAX
@_options.main.PASS_OPTIONS
def perform_inference(
    options: _options.main.Options,
    endpoint: str,
    weather_data: pathlib.Path,
    level: _options.data.Level,
    n_components: int,
    use_varimax: bool,
):
    """Use an AWS SageMaker Endpoint for inference."""
    options.exit_if_dry_run()
    df = _prepare_data(
        path=weather_data,
        level=level,
        n_components=n_components,
        use_varimax=use_varimax,
    )

    runtime = boto3.client("runtime.sagemaker")
    payload = df.to_json(orient="split")

    runtime_response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(runtime_response["Body"].read().decode())

    click.echo(f"Payload: {payload}")
    click.echo(f"Prediction: {result}")


def _prepare_data(
    path: pathlib.Path,
    level: _options.data.Level,
    n_components: int,
    use_varimax: bool,
) -> pd.DataFrame:
    """Run PCA on ECMWF IFS HRES data."""
    ds = xr.open_dataset(path).sel(level=level)["t"]

    pca = methods.spatio_temporal_pca(
        data=ds,
        algorithm=decomposition.PCA(n_components=n_components),
    )

    if use_varimax:
        result = pca.transform_with_varimax_rotation()
    else:
        result = pca.transform()

    return pd.DataFrame(result)
