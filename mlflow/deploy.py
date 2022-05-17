"""Based on example from
https://aws.amazon.com/blogs/machine-learning/managing-your-machine-learning-lifecycle-with-mlflow-and-amazon-sagemaker/  # noqa
i.e. the notebook
https://github.com/aws-samples/amazon-sagemaker-mlflow-fargate/blob/main/lab/3_deploy_model.ipynb  # noqa
"""
import argparse

import mlflow.sagemaker


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint-name", type=str, help="Name of the SageMaker endpoint."
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
            "The location, in URI format, of the MLflow model to deploy to "
            "SageMaker."
        ),
    )
    parser.add_argument(
        "--bucket", type=str, help="The S3 bucket to use for SageMaker."
    )
    parser.add_argument(
        "--role",
        type=str,
        help=(
            "Role (ARN) with access to the specified Docker image and S3 "
            "bucket contianer the MLflow model artifacts. Defaults to the "
            "current active role."
        ),
    )
    parser.add_argument(
        "--region",
        type=str,
        help=(
            "Region in which to deploy the model. Defaults to the current "
            "active region"
        ),
    )
    parser.add_argument(
        "--vpc-config",
        type=str,
        help="VPC configuration for the SageMaker instance.",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.m5.large",
        help="The Amazon SageMaker instance type",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of instances to run",
    )
    args = parser.parse_args()

    mlflow.sagemaker.deploy(
        app_name=args.endpoint_name,
        model_uri=args.model_uri,
        mode=mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE,
        image_url=args.image_uri,
        bucket=args.bucket,
        execution_role_arn=args.role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        region_name=args.region,
        # `archive=False`: Pre-existing SageMaker application resources that
        # become inactive will not be preserved.
        archive=False,
    )
