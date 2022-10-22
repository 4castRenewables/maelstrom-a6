import a6.cli.main as main


def test_deploy_model_to_sagemaker(runner):
    args = [
        "--dry-run",
        "deploy",
        "test-endpoint",
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

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
