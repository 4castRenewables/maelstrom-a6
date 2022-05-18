import lifetimes.cli.inference as inference


def test_parser():
    parser = inference.parser()
    args = parser.parse_args(
        [
            "--data",
            "test-path",
            "--variance-ratio",
            "1.0",
            "--use-varimax",
            "True",
        ]
    )

    assert args.data == "test-path"
    assert args.variance_ratio == 1.0
    assert args.use_varimax
