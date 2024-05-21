# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import argparse
import logging
import pathlib
from collections.abc import Iterator

ROOT_DIR = pathlib.Path(__file__).parent / "../../../"

logger = logging.getLogger(__name__)


class ExtendAction(argparse.Action):
    """Allows to pass multiple values for an argument as quoted strings.

    This is required to allow passing multiple arguments with MLflow,
    since MLflow puts given arguments into single-quoted strings.

    Examples
    --------
    1. `--option 1 2 3` yields `[1, 2, 3]`.
    2. `--option '1 2 3'` yields `[1, 2, 3]`.
    2. `--option "1 2 3"` yields `[1, 2, 3]`.

    """

    def __init__(self, type: type, *args, **kwargs):
        self._type = type
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        for value in values:
            # If `None` is given, don't append
            if _parse_str_or_none(value) is None:
                pass
            # If single- or double-quote in string, split by spaces and extend
            elif '"' in value or "'" in value:
                split = _split_multiple_quoted_arguments(value)
                items.extend(map(self._type, split))
            # If value is single value without quotes, append
            else:
                items.append(self._type(value))
        # Only set attribute if items given, else attribute should be ``None``
        if items:
            setattr(namespace, self.dest, items)


def _parse_str_or_none(value: str) -> str | None:
    if value.lower() == "none":
        return None
    return value


def _split_multiple_quoted_arguments(value: str) -> Iterator:
    for char in ["'", '"']:
        value = value.replace(char, "")
    # Split space- or comma-separated quoted list
    if " " in value:
        return value.split(" ")
    return iter([value])


class ExtendListAction(argparse.Action):
    """Allows to pass multiple tuple arguments.

    Examples
    --------
    1. `--option 1` yields `[1]`.
    2. `--option (1,2)` yields `[(1, 2)]`.
    3. `--option 1 --option (2,3)` yields `[1, (1, 2)]`.
    3. `--option '1 2'` yields `[1, 2]`.
    3. `--option '1 (2,3)'` yields `[1, (2, 3)]`.

    """

    def __init__(self, type: type, *args, **kwargs):
        self._type = type
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []

        # If default is a list, use a clean list to not
        # append to default.
        if items == self.default:
            items = []

        for value in values:
            split = _split_multiple_quoted_arguments(value)
            for element in split:
                # If , in values, assume tuple given
                if "," in element:
                    for char in ["(", ")"]:
                        element = element.replace(char, "")
                    try:
                        x, y = map(self._type, element.split(","))
                    except ValueError as e:
                        raise argparse.ArgumentTypeError(
                            f"Value for {self.dest} must be of type "
                            f"{self._type} or ({self._type},{self._type}), "
                            f"but {element} given"
                        ) from e
                    items.append((x, y))
                else:
                    items.append(self._type(element))
        # Only set attribute if items given, else attribute should be ``None``
        if not items:
            raise ValueError(
                f"Unable to extract any values for {self.dest} from {values}"
            )
        setattr(namespace, self.dest, items)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Implementation of DeepCluster-v2"
    )

    parser.add_argument(
        "--testing",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether current run is a testing run",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging (logging level DEBUG)",
    )

    parser.add_argument(
        "--use-cpu",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CPU for training",
    )
    parser.add_argument(
        "--use-nccl",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVIDIA NCCL backend for distributed computing",
    )
    parser.add_argument(
        "--enable-tracking",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable tracking to Mantik",
    )
    parser.add_argument(
        "--save-results",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save results as tensors every 20 epochs until epoch 100, "
            "then every 100 epochs, and last epoch"
        ),
    )
    parser.add_argument(
        "--plot-results",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Plot results every 20 epochs until epoch 100, "
            "then every 100 epochs, and last epoch"
        ),
    )

    # data parameters
    parser.add_argument(
        "--data-path",
        type=pathlib.Path,
        default=ROOT_DIR / "src/tests/data/deepclusterv2",
        help="Path to dataset repository",
    )
    parser.add_argument(
        "--use-mnist",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to use the MNIST dataset for testing purposes."
            ""
            "WARNING: If the dataset is not present in `data-path`, "
            "it will be downloaded! This may fail on the compute nodes of JSC."
        ),
    )
    parser.add_argument(
        "--use-imagenet",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use the ImageNet dataset.",
    )
    parser.add_argument(
        "--pattern",
        type=_parse_str_or_none,
        default=None,
        help=(
            "Pattern of the data files within the given data path."
            ""
            "If given, data loader for ``xarray.Dataset`` will be used."
        ),
    )
    parser.add_argument(
        "--drop-variables",
        type=str,
        default=None,
        nargs="+",
        action=ExtendAction,
        help="List of variables to drop from the dataset",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        nargs="+",
        action=ExtendAction,
        help=(
            "The levels to use from the dataset."
            ""
            "Per default all levels will be passed to the first layer"
            "of the CNN, where the levels are concatenated as channels. "
            "E.g. 2 quantities on 2 levels give 4 channels as input for "
            "the first layer."
        ),
    )
    parser.add_argument(
        "--select-dwd-area",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to select the area the DWD uses for Großwetterlagen."
            ""
            "The DWD uses an area of roughly 41.47°N-55.76°N and 0.0°E-19.66°E"
            ""
            "See https://www.dwd.de/DE/leistungen/wetterlagenklassifikation/beschreibung.html"  # noqa
        ),
    )
    parser.add_argument(
        "--parallel-loading",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable parallel loading",
    )

    # Transform parameters
    parser.add_argument(
        "--nmb-crops",
        type=int,
        nargs="+",
        default=[2],
        action=ExtendListAction,
        help="list of number of crops (example: 2 6)",
    )
    parser.add_argument(
        "--crops-for-assign",
        type=int,
        nargs="+",
        default=None,
        action=ExtendListAction,
        help=(
            "crop indices used for computing assignments (example for "
            "`--nmb-crops 2`: 0 1)"
        ),
    )
    parser.add_argument(
        "--size-crops",
        type=float,
        nargs="+",
        default=[0.65],
        action=ExtendListAction,
        help="Crops resolutions (example: 0.9 0.75)",
    )
    parser.add_argument(
        "--min-scale-crops",
        type=float,
        nargs="+",
        default=[0.08],
        action=ExtendListAction,
        help="argument in RandomResizedCrop (example: 0.14 0.05)",
    )
    parser.add_argument(
        "--max-scale-crops",
        type=float,
        nargs="+",
        default=[1.0],
        action=ExtendListAction,
        help="argument in RandomResizedCrop (example: 1. 0.14)",
    )

    # dcv2 specific params
    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="temperature parameter in training loss",
    )
    parser.add_argument(
        "--feat-dim", default=128, type=int, help="feature dimension"
    )
    parser.add_argument(
        "--nmb-prototypes",
        default=3,
        type=int,
        help="number of prototypes - it can be multihead",
    )
    parser.add_argument(
        "--nmb-clusters",
        default=2,
        type=int,
        help="number of clusters per prototype",
    )

    # optim parameters
    parser.add_argument(
        "--epochs", default=1, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch-size",
        default=2,
        type=int,
        help="batch size per gpu, i.e. how many unique instances per gpu",
    )
    parser.add_argument(
        "--base-lr", default=4.8, type=float, help="base learning rate"
    )
    parser.add_argument(
        "--final-lr", type=float, default=0.05, help="final learning rate"
    )
    parser.add_argument(
        "--freeze-prototypes-niters",
        default=3e6,
        type=int,
        help="freeze the prototypes during this many iterations from the start",
    )
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument(
        "--warmup-epochs", default=10, type=int, help="number of warmup epochs"
    )
    parser.add_argument(
        "--start-warmup",
        default=0.3,
        type=float,
        help="initial warmup learning rate",
    )

    # other parameters
    parser.add_argument(
        "--arch", default="resnet50", type=str, help="convnet architecture"
    )
    parser.add_argument(
        "--hidden-mlp",
        default=2048,
        type=int,
        help="hidden layer dimension in projection head",
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="number of data loading workers"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25,
        help="Save the model periodically",
    )
    parser.add_argument(
        "--sync-bn", type=str, default="pytorch", help="synchronize bn"
    )
    parser.add_argument(
        "--dump-path",
        type=pathlib.Path,
        default=ROOT_DIR / "dcv2_dump",
        help="experiment dump path for checkpoints and log",
    )
    parser.add_argument("--seed", type=int, default=31, help="seed")

    return parser
