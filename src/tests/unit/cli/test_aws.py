import lifetimes.cli.aws as aws


def test_sagemaker_deployment_args():
    parser = aws.create_sagemaker_deployment_parser()
    args = parser.parse_args(
        [
            "--endpoint-name",
            "test",
            "--image-uri",
            "test-ecr-image",
            "--model-uri",
            "models:/test-model/1",
            "--bucket",
            "test-s3-bucket",
            "--role",
            "test-role-arn",
            "--region",
            "test-region",
            "--vpc-config",
            '{"TestKey": "test-value"}',
            "--instance-type",
            "test-instance-type",
            "--instance-count",
            "2",
        ]
    )

    assert args.endpoint_name == "test"
    assert args.image_uri == "test-ecr-image"
    assert args.model_uri == "models:/test-model/1"
    assert args.bucket == "test-s3-bucket"
    assert args.role == "test-role-arn"
    assert args.region == "test-region"
    assert args.vpc_config == '{"TestKey": "test-value"}'
    assert args.instance_type == "test-instance-type"
    assert args.instance_count == 2


def test_sagemaker_inference_args():
    parser = aws.create_sagemaker_inference_parser()
    args = parser.parse_args(["--endpoint-name", "test"])

    assert args.endpoint_name == "test"
