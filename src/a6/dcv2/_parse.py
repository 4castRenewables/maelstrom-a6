# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import argparse
import logging
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent / "../../../"

logger = logging.getLogger(__name__)


class ExtendAction(argparse.Action):
    def __init__(self, type: type, *args, **kwargs):
        self._type = type
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        for value in values:
            # If `None` is given, don't append
            if parse_str_or_none(value) is None:
                pass
            # If single- or double-quote in string, split by spaces and extend
            elif '"' in value or "'" in value:
                value = value.replace('"', "").replace("'", "")
                # Split space-separate quoted list
                result = value.split(" ")
                items.extend(map(self._type, result))
            # If value is single value without quotes, append
            else:
                items.append(self._type(value))
        # Only set attribute if items given, else attribute should be ``None``
        if items:
            setattr(namespace, self.dest, items)


def parse_str_or_none(value: str) -> str | None:
    if value.lower() == "none":
        return None
    return value


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Implementation of DeepCluster-v2"
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
        "--enable-tracking",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable tracking to Mantik",
    )

    # data parameters
    parser.add_argument(
        "--data-path",
        type=pathlib.Path,
        default=ROOT_DIR / "src/tests/data/deepclusterv2",
        help="Path to dataset repository",
    )
    parser.add_argument(
        "--pattern",
        type=parse_str_or_none,
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
        action=ExtendAction,
        help="list of number of crops (example: [2, 6])",
    )
    parser.add_argument(
        "--size-crops",
        type=float,
        nargs="+",
        default=[0.75],
        action=ExtendAction,
        help="Crops resolutions (example: [0.9, 0.75])",
    )
    parser.add_argument(
        "--min-scale-crops",
        type=float,
        nargs="+",
        default=[0.14],
        action=ExtendAction,
        help="argument in RandomResizedCrop (example: [0.14, 0.05])",
    )
    parser.add_argument(
        "--max-scale-crops",
        type=float,
        nargs="+",
        default=[1],
        action=ExtendAction,
        help="argument in RandomResizedCrop (example: [1., 0.14])",
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
        "--final-lr", type=float, default=0, help="final learning rate"
    )
    parser.add_argument(
        "--freeze-prototypes-niters",
        default=1e10,
        type=int,
        help="freeze the prototypes during this many iterations from the start",
    )
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument(
        "--warmup-epochs", default=10, type=int, help="number of warmup epochs"
    )
    parser.add_argument(
        "--start-warmup",
        default=0,
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
        "--syncbn-process-group-size",
        type=int,
        default=8,
        help=(
            "see"
            "https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67"  # noqa: E501
        ),
    )
    parser.add_argument(
        "--dump-path",
        type=pathlib.Path,
        default=ROOT_DIR / "dcv2_dump",
        help="experiment dump path for checkpoints and log",
    )
    parser.add_argument("--seed", type=int, default=31, help="seed")

    return parser
