# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import contextlib
import logging
import math
import shutil
import socket
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed.elastic.multiprocessing.errors as errors
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import a6.datasets as datasets
import a6.dcv2._checkpoints as _checkpoints
import a6.dcv2._initialization as _initialization
import a6.dcv2._parse as _parse
import a6.dcv2._settings as _settings
import a6.dcv2.cluster as cluster
import a6.dcv2.dataset as dataset
import a6.dcv2.logs as logs
import a6.dcv2.models as models
import a6.dcv2.train as train
import a6.features as features
import a6.utils as utils
import a6.utils.mantik as mantik
import mlflow


@errors.record
def train_dcv2(raw_args: list[str] | None = None):
    args = _parse.create_argparser().parse_args(raw_args)
    settings = _settings.Settings.from_args_and_env(args)

    logger, training_stats = _initialization.initialize_logging(
        settings, columns=("epoch", "loss")
    )

    with setup_distributed(settings):
        _train(
            settings=settings,
            logger=logger,
            training_stats=training_stats,
        )


@contextlib.contextmanager
def setup_distributed(settings: _settings.Settings) -> None:
    utils.distributed.setup(settings.distributed, seed=settings.data.seed)

    if utils.distributed.is_primary_device():
        start = time.time()

        utils.logging.log_env_vars()

        stdout = utils.slurm.get_stdout_file()
        stderr = utils.slurm.get_stderr_file()

        if settings.enable_tracking:
            mantik.call_mlflow_method(mlflow.start_run)

            if utils.slurm.is_slurm_job():
                mantik.call_mlflow_method(
                    mlflow.log_params,
                    utils.slurm.get_slurm_env_vars(),
                )

    yield

    if utils.distributed.is_primary_device():
        logging.info("All done!")
        runtime = time.time() - start
        logging.info("Total runtime (s): %s", runtime)

        if settings.enable_tracking:
            mantik.call_mlflow_method(
                mlflow.log_metric, "total_runtime_s", runtime
            )

            _log_stdout_stderr(stdout=stdout, stderr=stderr)

            # The following files are saved to disk in `a6.dcv2.cluster.py`
            mantik.call_mlflow_method(
                mlflow.log_artifacts, settings.dump.results
            )

            mantik.call_mlflow_method(mlflow.end_run)

    utils.distributed.destroy()


def _train(
    settings: _settings.Settings,
    logger: logging.Logger,
    training_stats: logs.Stats,
):
    logger.info(
        (
            "Starting training on host %s (node %s), global rank %s, "
            "local rank %s (torch.distributed.get_rank()=%s)"
        ),
        socket.gethostname(),
        settings.distributed.node_id,
        settings.distributed.global_rank,
        settings.distributed.local_rank,
        torch.distributed.get_rank(),
    )

    # build data
    train_dataset = _create_dataset(settings, logger=logger)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=settings.model.batch_size,
        num_workers=settings.data.workers,
        pin_memory=True,
        # Do not drop last batch to include all samples when clustering.
        drop_last=False,
    )
    logger.info("Building data done with %s images loaded", len(train_dataset))

    device = utils.distributed.get_device(settings.distributed)

    # build model
    model = models.__dict__[settings.model.architecture](
        normalize=True,
        in_channels=train_dataset.n_channels,
        hidden_mlp=settings.model.hidden_mlp,
        output_dim=settings.model.feature_dimensions,
        nmb_prototypes=settings.model.nmb_prototypes,
        device=device,
    )
    # synchronize batch norm layers
    match settings.model.sync_bn:
        case _settings.SyncBn.PYTORCH:
            if settings.distributed.use_cpu:
                pass
            else:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        case _settings.SyncBn.APEX:
            if utils.distributed.is_multi_gpu():
                import apex
                from apex.parallel.LARC import LARC

                # with apex syncbn we sync bn per group because it speeds up
                # computation compared to global syncbn
                process_group = apex.parallel.create_syncbn_process_group(
                    settings.model.syncbn_process_group_size
                )
                model = apex.parallel.convert_syncbn_model(
                    model, process_group=process_group
                )
            else:
                logger.warning(
                    "Sync batch norm 'apex' defined, but training is performed "
                    "on single GPU, hence using native PyTorch"
                )
        case _:
            raise NotImplementedError(
                f"SyncBn {settings.model.sync_bn} not implemented"
            )

    # Copy model to GPU
    model = model.to(device)

    if utils.distributed.is_primary_device():
        logger.info(model)

    logger.info("Building model done")

    # build optimizer
    # Should be done after moving the model to GPU
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=settings.model.base_lr,
        momentum=0.9,
        weight_decay=settings.model.wd,
    )

    if settings.model.sync_bn == _settings.SyncBn.APEX:
        if utils.distributed.is_multi_gpu():
            optimizer = LARC(
                optimizer=optimizer, trust_coefficient=0.001, clip=False
            )
        else:
            logger.warning(
                "Sync batch norm 'apex' defined, but training is performed "
                "on single GPU, hence LARC is disabled"
            )

    warmup_lr_schedule = np.linspace(
        settings.model.start_warmup,
        settings.model.base_lr,
        len(train_loader) * settings.model.warmup_epochs,
    )
    iters = np.arange(
        len(train_loader)
        * (settings.model.epochs - settings.model.warmup_epochs)
    )
    cosine_lr_schedule = np.array(
        [
            settings.model.final_lr
            + 0.5
            * (settings.model.base_lr - settings.model.final_lr)
            * (
                1
                + math.cos(
                    math.pi
                    * t
                    / (
                        len(train_loader)
                        * (settings.model.epochs - settings.model.warmup_epochs)
                    )
                )
            )
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done")

    # wrap model
    if not settings.distributed.use_cpu:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[settings.distributed.local_rank],
            find_unused_parameters=True,
        )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    _checkpoints.restart_from_checkpoint(
        settings.dump.path / "checkpoint.pth.tar",
        settings=settings,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    # build the memory bank
    mb_path = settings.dump.path / f"mb-{settings.distributed.global_rank}.pth"

    if mb_path.is_file():
        mb_ckp = torch.load(mb_path)
        local_memory_index = mb_ckp["local_memory_index"]
        local_memory_embeddings = mb_ckp["local_memory_embeddings"]
    else:
        local_memory_index, local_memory_embeddings = cluster.init_memory(
            dataloader=train_loader,
            model=model,
            settings=settings,
            device=device,
        )

    cudnn.benchmark = True
    for epoch in range(start_epoch, settings.model.epochs):
        # train the network for one epoch
        if utils.distributed.is_primary_device():
            logger.info("============ Starting epoch %i ============", epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores, local_memory_index, local_memory_embeddings = train.train(
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            schedule=lr_schedule,
            local_memory_index=local_memory_index,
            local_memory_embeddings=local_memory_embeddings,
            settings=settings,
            device=device,
        )
        training_stats.update(scores)

        # save checkpoints
        if utils.distributed.is_primary_device():
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                settings.dump.path / "checkpoint.pth.tar",
            )
            if (
                epoch % settings.dump.checkpoint_freq == 0
                or epoch == settings.model.epochs - 1
            ):
                shutil.copyfile(
                    settings.dump.path / "checkpoint.pth.tar",
                    settings.dump.checkpoints / f"checkpoint-epoch-{epoch}.pth",
                )
        torch.save(
            {
                "local_memory_embeddings": local_memory_embeddings,
                "local_memory_index": local_memory_index,
            },
            mb_path,
        )


def _create_dataset(
    settings: _settings.Settings, logger: logging.Logger
) -> dataset.Base:
    if settings.data.pattern is not None:
        # If a data pattern is given, it is assumed that the
        # given data path is a folder with netCDF files.
        logger.warning("Assuming `xarray.Dataset` from netCDF files")
        coordinates = datasets.coordinates.Coordinates()
        variables = datasets.variables.Model()
        drop_variables = settings.data.drop_variables or []

        preprocessing = (
            features.methods.weighting.weight_by_latitudes(
                latitudes=coordinates.latitude,
                use_sqrt=True,
            )
            >> features.methods.geopotential.calculate_geopotential_height(
                variables=variables,
            )
            >> features.methods.variables.drop_variables(
                names=[variables.z] + drop_variables
            )
        )

        if settings.data.select_dwd_area:
            preprocessing = (
                preprocessing
                >> datasets.methods.select.select_dwd_area(
                    coordinates=coordinates
                )
            )

        ds = datasets.Era5(
            path=settings.data.path,
            pattern=settings.data.pattern,
            preprocessing=preprocessing,
            parallel_loading=settings.data.parallel_loading,
        ).to_xarray(levels=settings.data.levels)
        return dataset.MultiCropXarrayDataset(
            data_path=settings.data.path,
            dataset=ds,
            nmb_crops=settings.preprocessing.nmb_crops,
            size_crops=settings.preprocessing.size_crops,
            min_scale_crops=settings.preprocessing.min_scale_crops,
            max_scale_crops=settings.preprocessing.max_scale_crops,
            return_index=True,
        )
    return dataset.MultiCropDataset(
        data_path=settings.data.path,
        nmb_crops=settings.preprocessing.nmb_crops,
        size_crops=settings.preprocessing.size_crops,
        min_scale_crops=settings.preprocessing.min_scale_crops,
        max_scale_crops=settings.preprocessing.max_scale_crops,
        return_index=True,
    )


def _log_stdout_stderr(stdout: str | None, stderr: str | None) -> None:
    if stdout is not None:
        mantik.call_mlflow_method(mlflow.log_artifact, stdout)
    if stderr is not None:
        mantik.call_mlflow_method(mlflow.log_artifact, stderr)
