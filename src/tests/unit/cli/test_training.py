import lifetimes.cli.training as training


def test_parser():
    parser = training.create_parser()
    args = parser.parse_args(
        [
            "--data",
            "test-path",
            "--variance-ratios",
            "1.0",
            "0.95",
            "--n-clusters",
            "1",
            "2",
            "--use-varimax",
            "True",
            "False",
        ]
    )

    assert args.data == "test-path"
    assert args.variance_ratios == [1.0, 0.95]
    assert args.n_clusters == [1, 2]
    assert args.use_varimax == [True, False]
