# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import argparse
import logging
import pathlib
from typing import Any

ROOT_DIR = pathlib.Path(__file__).parent / "../../../"

logger = logging.getLogger(__name__)


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
        type=str,
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
        help="List of variables to drop from the dataset",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=500,
        help=(
            "The level to use from the dataset."
            ""
            "Training with multiple levels is not yet supported."
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

    # Transform parameters
    parser.add_argument(
        "--nmb-crops",
        type=int,
        default=[2],
        nargs="+",
        help="list of number of crops (example: [2, 6])",
    )
    parser.add_argument(
        "--size-crops",
        type=float,
        default=[0.75],
        nargs="+",
        help="Crops resolutions (example: [0.9, 0.75])",
    )
    parser.add_argument(
        "--min-scale-crops",
        type=float,
        default=[0.14],
        nargs="+",
        help="argument in RandomResizedCrop (example: [0.14, 0.05])",
    )
    parser.add_argument(
        "--max-scale-crops",
        type=float,
        default=[1],
        nargs="+",
        help="argument in RandomResizedCrop (example: [1., 0.14])",
    )

    # dcv2 specific params
    parser.add_argument(
        "--crops-for-assign",
        type=int,
        nargs="+",
        default=[0, 1],
        help="list of crops id used for computing assignments",
    )
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

    # dist parameters
    parser.add_argument(
        "--node-id",
        default=0,
        type=int,
        help=("Node ID of the node that the process runs on."),
    )
    parser.add_argument(
        "--host",
        default=None,
        type=str,
        help=(
            "URL used to set up distributed training. "
            "It is set automatically and should not be passed as argument. "
            "See https://pytorch.org/docs/stable/distributed.html"
        ),
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://",
        type=str,
        help=(
            "URL used to set up distributed training. "
            "It is set automatically and should not be passed as argument. "
            "See https://pytorch.org/docs/stable/distributed.html"
        ),
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help=(
            "Total number of processes. "
            "This may e.g. be the total number of available GPU devices. "
            "It is set automatically and should not be passed as argument."
        ),
    )
    parser.add_argument(
        "--global-rank",
        default=0,
        type=int,
        help=(
            "Global rank of this process. "
            "It is set automatically and should not be passed as argument."
        ),
    )
    parser.add_argument(
        "--local-rank",
        default=0,
        type=int,
        help=(
            "This argument is passed by ``torch.distributed.launch``. "
            "See https://pytorch.org/docs/stable/distributed.html#launch-utility"  # noqa: E501
        ),
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
        "--workers", default=1, type=int, help="number of data loading workers"
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


def overwrite_arg(args, attribute: str, value: Any) -> None:
    prev = getattr(args, attribute)
    logger.warning("Overwriting args.%s=%s with %s", attribute, prev, value)
    setattr(args, attribute, value)
