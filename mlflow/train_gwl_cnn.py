# invoke using apptainer run --cleanenv --nv -B /p/home/jusers/$USER/juwels/code/a6/:/opt/a6 /p/project/deepacf/maelstrom/emmerich1/a6-cuda.sif python /opt/a6/mlflow/train_gwl_cnn.py
import argparse
import pathlib

import a6


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weather-data",
        type=pathlib.Path,
        help="Path to the weather dataset",
    )
    parser.add_argument(
        "--gwl-data",
        type=pathlib.Path,
        help="Path to the GWL data",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="The CNN model architecture to use for training",
        choices=["cnn"] + [choice.value for choice in a6.models.resnet.Architecture],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--select-dwd-area",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to select the DWD area from the input data",
    )
    parser.add_argument(
        "--enable-tracking",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to track the training process to Mantik",
    )
    args = parser.parse_args()

    a6.entry.gwl.main(
        data_path=args.weather_data,
        gwl_path=args.gwl_data,
        epochs=args.epochs,
        architecture=args.architecture,
        select_dwd_area=args.select_dwd_area,
        testing=False,
        log_to_mlflow=args.enable_tracking,
    )
