"""Based on example from
https://aws.amazon.com/blogs/machine-learning/managing-your-machine-learning-lifecycle-with-mlflow-and-amazon-sagemaker/  # noqa
i.e. the notebook
https://github.com/aws-samples/amazon-sagemaker-mlflow-fargate/blob/main/lab/3_deploy_model.ipynb  # noqa
"""
import json

import click

import a6.cli.arguments as arguments
import a6.cli.main as main
import a6.cli.options as _options

_DEFAULT_SAGEMAKER_INSTANCE_TYPE = "ml.m5.large"


@main.cli.command("deploy")
@arguments.sagemaker.ENDPOINT
@click.option(
    "--image-uri",
    type=str,
    required=True,
    help=(
        "URL of the ECR-hosted Docker image the model should be deployed with."
    ),
)
@click.option(
    "--model-uri",
    type=str,
    required=True,
    help="""
        The location of the MLflow model in the MLflow Model Registry.

        \b
        Example: `models:/my-model/1`, where 1 is the model version.
        For more details see
        https://www.mlflow.org/docs/latest/concepts.html#artifact-locations
    """,
)
@click.option(
    "--bucket",
    type=str,
    required=True,
    help="Name the S3 bucket where SageMaker stores temporary model data.",
)
@click.option(
    "--role",
    type=str,
    required=True,
    help="""
        Role (ARN) with access to the specified Docker image and S3
        bucket container the MLflow model artifacts.

        \b
        Required permissions:
            - AmazonS3ReadOnlyAccess
            - AmazonSageMakerFullAccess
    """,
)
@click.option(
    "--region",
    type=str,
    required=True,
    help="Region in which to deploy the model.",
)
@click.option(
    "--vpc-config",
    type=str,
    required=True,
    help="""
        VPC JSON config with the required VPC IDs for the SageMaker instance.

        \b
        Required fields are:
            - `SecurityGroupIds`: ID of the security group of the
              MLflow VPC.
            - `Subnets`: ID of the private subnet of the MLflow VPC.

        \b
        Example:
        ```json
        {
          "SecurityGroupIds": ["sg-123abc"],
          "Subnets": ["subnet-123abc"]
        }
        ```
    """,
)
@click.option(
    "--instance-type",
    type=str,
    required=False,
    default=_DEFAULT_SAGEMAKER_INSTANCE_TYPE,
    show_default=True,
    help="""
        The SageMaker instance type.

        \b
        See https://aws.amazon.com/sagemaker/pricing
    """,
)
@click.option(
    "--instance-count",
    type=int,
    default=1,
    required=False,
    help="Number of instances to run. Defaults to 1.",
)
@_options.main.PASS_OPTIONS
def deploy_model_to_sagemaker(
    options: _options.main.Options,
    endpoint: str,
    image_uri: str,
    model_uri: str,
    bucket: str,
    role: str,
    region: str,
    vpc_config: str,
    instance_type: str,
    instance_count: int,
):
    """Deploy a registered MLflow model to an AWS SageMaker Endpoint."""
    import mlflow.sagemaker

    options.exit_if_dry_run()
    mlflow.sagemaker.deploy(
        app_name=endpoint,
        model_uri=model_uri,
        mode=mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE,
        image_url=image_uri,
        bucket=bucket,
        execution_role_arn=role,
        instance_type=instance_type,
        instance_count=instance_count,
        region_name=region,
        vpc_config=json.loads(vpc_config),
        # `archive=False`: Pre-existing SageMaker application resources that
        # become inactive will not be preserved.
        archive=False,
    )
