import argparse

DEFAULT_SAGEMAKER_INSTANCE_TYPE = "ml.m5.large"


def deployment_args() -> argparse.ArgumentParser:
    """Parse args necessary for AWS SageMaker deployment of an MLflow model.

    Returns
    -------
    parser : argparse.ArgumentParser
        Arguments can be retrieved via `parser.parse_args()`.
        Additional arguments can be added via `parser.add_argument()`.
        Default arguments that will be available as attributes after parsing:
            endpoint_name : str
                Name of the SageMaker Endpoint.
            image_uri : str
                URI to the Docker image in the ECR.
            model_uri : str
                URI to the registered model.
                Example: `models:/my-model/1`, where `1` is the model version.
            bucket : str
                Name of the S3 bucket that is the MLflow artifact bucket.
            role : str
                AWS role with the necessary permissions for SageMaker.
                Required permissions are:
                    - AmazonS3ReadOnlyAccess
                    - AmazonSageMakerFullAccess
            region : str
                AWS region to deploy the SageMaker Endpoint.
            vpc_config : str
                VPC JSON config with the required VPC IDs.
                Required fields are:
                    - `SecurityGroupIds`: ID of the security group of the
                      MLflow VPC.
                    - `Subnets`: ID of the private subnet of the MLflow VPC.
                Example:
                ```JSON
                {
                  "SecurityGroupIds": ["sg-123abc"],
                  "Subnets": ["subnet-123abc"]
                }
                ```
            instance_type : str
                SageMaker resource instance type.
                See https://aws.amazon.com/sagemaker/pricing/
            instance_count : int
                Number of SageMaker instances to run.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint-name", type=str, help="Name of the SageMaker Endpoint."
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        help=(
            "URL of the ECR-hosted Docker image the model should be deployed "
            "into."
        ),
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        help=(
            "The location of the MLflow model in the MLflow Model Registry to "
            "deploy to SageMaker. Example: `models:/my-model/1`, where 1 is "
            "the model version."
        ),
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="Name of the MLflow Artifact Storage S3 bucket.",
    )
    parser.add_argument(
        "--role",
        type=str,
        help=(
            "Role (ARN) with access to the specified Docker image and S3 "
            "bucket container the MLflow model artifacts."
        ),
    )
    parser.add_argument(
        "--region",
        type=str,
        help=("Region in which to deploy the model."),
    )
    parser.add_argument(
        "--vpc-config",
        type=str,
        help="VPC configuration for the SageMaker instance.",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default=DEFAULT_SAGEMAKER_INSTANCE_TYPE,
        help=(
            "The Amazon SageMaker instance type. Defaults to "
            f"`{DEFAULT_SAGEMAKER_INSTANCE_TYPE}`."
        ),
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of instances to run. Defaults to 1.",
    )
    return parser


def inference_args() -> argparse.ArgumentParser:
    """Parse args necessary for AWS SageMaker inference via en Endpoint.

    Returns
    -------
    parser : argparse.ArgumentParser
        Arguments can be retrieved via `parser.parse_args()`.
        Additional arguments can be added via `parser.add_argument()`.
        Default arguments that will be available as attributes after parsing:
            endpoint_name : str
                Name of the SageMaker Endpoint to use for inference.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint-name",
        type=str,
        help="Name of the SageMaker Endpoint to use for inference.",
    )
    return parser
