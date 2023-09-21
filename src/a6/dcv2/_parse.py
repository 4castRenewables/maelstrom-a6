# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import argparse
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent / "../../../"


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Implementation of DeepCluster-v2"
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
        help="path to dataset repository",
    )
    parser.add_argument(
        "--nmb-crops",
        type=int,
        default=[2],
        nargs="+",
        help="list of number of crops (example: [2, 6])",
    )
    parser.add_argument(
        "--size-crops",
        type=int,
        default=[96],
        nargs="+",
        help="crops resolutions (example: [224, 96])",
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
        default=[2, 2, 2],
        type=int,
        nargs="+",
        help="number of prototypes - it can be multihead",
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
        "--dist-url",
        default="tcp://",
        type=str,
        help=(
            "url used to set up distributed"
            "training; see https://pytorch.org/docs/stable/distributed.html"
        ),
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""",
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int,
        help=(
            "rank of this process:"
            "it is set automatically and should not be passed as argument"
        ),
    )
    parser.add_argument(
        "--local-rank",
        default=0,
        type=int,
        help="this argument is not used and should be ignored",
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
