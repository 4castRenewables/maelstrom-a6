import lifetimes.cli.inference as inference


def test_parser():
    parser = inference.create_parser()
    args = parser.parse_args(
        [
            "--data",
            "test-path",
            "--n-components",
            "1",
            "--use-varimax",
            "True",
        ]
    )

    assert args.data == "test-path"
    assert args.n_components == 1
    assert args.use_varimax
