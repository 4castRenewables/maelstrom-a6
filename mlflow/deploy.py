"""Based on example from
https://aws.amazon.com/blogs/machine-learning/managing-your-machine-learning-lifecycle-with-mlflow-and-amazon-sagemaker/  # noqa
i.e. the notebook
https://github.com/aws-samples/amazon-sagemaker-mlflow-fargate/blob/main/lab/3_deploy_model.ipynb  # noqa
"""
import lifetimes.cli.aws as aws

import mlflow.sagemaker


if __name__ == "__main__":

    parser = aws.create_sagemaker_deployment_parser()
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
