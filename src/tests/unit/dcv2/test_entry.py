import a6.dcv2.entry as entry


def test_train_dcv2():
    # Train first epoc
    raw_args_1 = ["--use-cpu", "--epochs", "1"]
    entry.train_dcv2(raw_args_1)

    # Train second epoch to restore from dump path
    raw_args_2 = ["--use-cpu", "--epochs", "2"]
    entry.train_dcv2(raw_args_2)
