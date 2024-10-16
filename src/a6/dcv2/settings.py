import dataclasses
import enum
import pathlib
from typing import Any

import a6.models.resnet_dcv2 as resnet_dcv2
import a6.utils as utils


class SyncBn(enum.Enum):
    PYTORCH = "pytorch"
    APEX = "apex"


@dataclasses.dataclass(frozen=True)
class Data:
    path: pathlib.Path
    use_mnist: bool
    use_imagenet: bool
    pattern: str | None
    drop_variables: bool
    levels: list[int] | None
    select_dwd_area: bool
    parallel_loading: bool
    workers: int
    seed: int


@dataclasses.dataclass(frozen=True)
class Preprocessing:
    nmb_crops: list[int]
    size_crops: list[float]
    min_scale_crops: list[float]
    max_scale_crops: list[float]


@dataclasses.dataclass(frozen=True)
class Model:
    architecture: resnet_dcv2.Architecture
    hidden_mlp: int
    feature_dimensions: int
    nmb_prototypes: list[int]
    nmb_clusters: list[int]
    crops_for_assign: list[int]
    drop_last: bool
    temperature: float
    epochs: int
    batch_size: int
    base_lr: float
    final_lr: float
    sync_bn: SyncBn
    syncbn_process_group_size: int
    freeze_prototypes_niters: int
    wd: float
    warmup_epochs: int
    start_warmup: float


@dataclasses.dataclass(frozen=True)
class Dump:
    checkpoint_freq: int
    path: pathlib.Path
    checkpoints: pathlib.Path
    results: pathlib.Path
    plots: pathlib.Path
    tensors: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Settings:
    testing: bool
    verbose: bool
    enable_tracking: bool
    save_results: bool
    plot_results: bool
    distributed: utils.distributed.Properties
    data: Data
    preprocessing: Preprocessing
    model: Model
    dump: Dump

    def __post_init__(self):
        if self.data.use_mnist and self.data.use_imagenet:
            raise ValueError(
                "Both MNIST and ImageNet dataset usage enabled, "
                "please choose one"
            )

    @classmethod
    def from_args_and_env(cls, args) -> "Settings":
        env_vars = utils.distributed.get_and_set_required_env_vars()
        dump_checkpoints = args.dump_path / "checkpoints"
        dump_results = args.dump_path / "results"
        dump_plots = dump_results / "plots"
        dump_tensors = dump_results / "tensors"
        return cls(
            testing=args.testing,
            verbose=args.verbose,
            enable_tracking=args.enable_tracking,
            save_results=args.save_results,
            plot_results=args.plot_results,
            distributed=utils.distributed.Properties.from_env(
                use_cpu=args.use_cpu,
                use_nccl=args.use_nccl,
                verbose_logging=args.verbose,
                enable_tracking=args.enable_tracking,
                logs_filepath=args.dump_path / "train.log",
            ),
            data=Data(
                path=args.data_path,
                use_mnist=args.use_mnist,
                use_imagenet=args.use_imagenet,
                pattern=args.pattern,
                drop_variables=args.drop_variables,
                levels=args.levels,
                select_dwd_area=args.select_dwd_area,
                parallel_loading=args.parallel_loading,
                workers=args.workers,
                seed=args.seed,
            ),
            preprocessing=Preprocessing(
                nmb_crops=args.nmb_crops,
                size_crops=args.size_crops,
                min_scale_crops=args.min_scale_crops,
                max_scale_crops=args.max_scale_crops,
            ),
            model=Model(
                architecture=resnet_dcv2.Architecture(args.arch),
                hidden_mlp=args.hidden_mlp,
                feature_dimensions=args.feat_dim,
                # ``nmb_prototypes`` is a list with the number of clusters
                # (``K``) to use for the K-means. The length of the list
                # (``nmb_clusters``) defines how many times the clustering is
                # performed.
                nmb_prototypes=[
                    args.nmb_clusters for _ in range(args.nmb_prototypes)
                ],
                # ``crops_for_assign`` is the indexes of the crops used for
                # clustering and depends on ``args.nmb_crops``.
                # E.g. ``args.nmb_crops = [2, 4]`` produces 6 inputs
                # to the clustering. Hence, ``crops_for_assign`` must be
                # ``[0, 1, 2, ..., 5]``, which are the indexes of
                # ``nmb_crops``.
                crops_for_assign=args.crops_for_assign
                or [i for i in range(sum(args.nmb_crops))],
                # Do not drop last batch to include all samples when clustering.
                drop_last=False,
                temperature=args.temperature,
                nmb_clusters=args.nmb_clusters,
                epochs=args.epochs,
                batch_size=args.batch_size,
                base_lr=args.base_lr,
                final_lr=args.final_lr,
                sync_bn=SyncBn(args.sync_bn),
                # See https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67"  # noqa: E501
                syncbn_process_group_size=(
                    env_vars.world_size // utils.slurm.get_number_of_nodes()
                ),
                freeze_prototypes_niters=args.freeze_prototypes_niters,
                wd=args.wd,
                warmup_epochs=args.warmup_epochs,
                start_warmup=args.start_warmup,
            ),
            dump=Dump(
                checkpoint_freq=args.checkpoint_freq,
                path=args.dump_path,
                checkpoints=dump_checkpoints,
                results=dump_results,
                plots=dump_plots,
                tensors=dump_tensors,
            ),
        )

    def to_dict(self) -> dict:
        as_dict = dataclasses.asdict(self)
        return _to_strings(as_dict)


def _to_strings(d: dict) -> dict:
    return {
        key: _to_strings(value) if isinstance(value, dict) else _convert(value)
        for key, value in d.items()
    }


def _convert(value: Any | pathlib.Path) -> Any:
    if isinstance(value, pathlib.Path):
        return value.absolute().as_posix()
    if isinstance(value, enum.Enum):
        return str(value)
    return value
