# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import contextlib
import io
import logging
import math
import os
import socket
import time

import mantik.mlflow
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed.elastic.multiprocessing.errors as errors
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import a6.datasets as datasets
import a6.dcv2.cluster as cluster
import a6.dcv2.initialization as _initialization
import a6.dcv2.parse as _parse
import a6.dcv2.settings as _settings
import a6.dcv2.stats as stats
import a6.dcv2.train as train
import a6.models as models
import a6.utils as utils

_timer = utils.benchmark.import_deep500()

@errors.record
def run_benchmark(raw_args: list[str] | None = None):
    args = _parse.create_argparser().parse_args(raw_args)
    settings = _settings.Settings.from_args_and_env(args)

    logger, training_stats = _initialization.initialize_logging(
        settings, columns=("epoch", "loss")
    )

    hardware = "nvidia" if "CUDA_VERSION" in os.environ else "amd"
    logger.info("Assuming %s GPU hardware", hardware)

    energy_profiler = utils.energy.get_energy_profiler(hardware)

    with (
        setup_distributed(settings=settings, logger=logger),
        energy_profiler() as measured_scope,
    ):
        _train(
            settings=settings,
            logger=logger,
            training_stats=training_stats,
        )

    with open("Energy-integrated.txt", "a") as f:
        try:
            measured_scope.df.to_csv("Energy.csv")
        except AttributeError as e:
            raise RuntimeError(
                "Results DataFrame of energy profiler is only available "
                "after exiting the context"
            ) from e

        logger.info("Energy-per-GPU-list:")
        max_power = (
            measured_scope.df.loc[
                :, (measured_scope.df.columns != "timestamps")
            ]
            .max()
            .max()
        )
        logger.info("Max Power: %.2f W", max_power)

        max_agg_power = (
            measured_scope.df.loc[
                :, (measured_scope.df.columns != "timestamps")
            ]
            .sum(axis=1)
            .max()
        )
        logger.info("Max Aggregate Power: %.2f W", max_agg_power)

        mean_agg_power = (
            measured_scope.df.loc[
                :, (measured_scope.df.columns != "timestamps")
            ]
            .sum(axis=1)
            .mean()
        )
        logger.info("Mean Aggregate Power: %.2f W", mean_agg_power)

        if isinstance(measured_scope, utils.energy.GetAMDPower):
            energy_int, energy_cnt = measured_scope.energy()
            logger.info("Integrated Total Energy: %.2f Wh", np.sum(energy_int))
            logger.info("Counter Total Energy: %.2f Wh", np.sum(energy_cnt))

            f.write(f"integrated: {energy_int}")
            f.write(f"from counter: {energy_cnt}")
        else:
            energy_int = measured_scope.energy()
            logger.info("Integrated Total Energy: %.2f Wh", np.sum(energy_int))
            f.write(f"integrated: {energy_int}")


@errors.record
def train_dcv2(raw_args: list[str] | None = None):
    args = _parse.create_argparser().parse_args(raw_args)
    settings = _settings.Settings.from_args_and_env(args)

    logger, training_stats = _initialization.initialize_logging(
        settings, columns=["epoch", "loss"]
    )

    with setup_distributed(settings=settings, logger=logger):
        _train(
            settings=settings,
            logger=logger,
            training_stats=training_stats,
        )


@contextlib.contextmanager
def setup_distributed(
    settings: _settings.Settings, logger: logging.Logger
) -> None:
    if utils.distributed.is_primary_device():
        utils.logging.log_env_vars()

    utils.distributed.setup(settings.distributed, seed=settings.data.seed)

    if utils.distributed.is_primary_device():
        start = time.time()

        if settings.enable_tracking:
            stdout = utils.slurm.get_stdout_file()
            stderr = utils.slurm.get_stderr_file()
            mantik.mlflow.start_run()

            if utils.slurm.is_slurm_job():
                mantik.mlflow.log_params(utils.slurm.get_slurm_env_vars())

    yield

    if utils.distributed.is_primary_device():
        logger.info("All done!")
        runtime = time.time() - start
        logger.info("Total runtime: %s s", runtime)

        if settings.enable_tracking:
            mantik.mlflow.log_metric("total_runtime_s", runtime)

            _log_stdout_stderr(stdout=stdout, stderr=stderr)

            # The following files are saved to disk in `a6.dcv2.cluster.py`
            # NOTE: Only log plots due to large size of other files
            mantik.mlflow.log_artifacts(settings.dump.plots)

            mantik.mlflow.end_run()

    utils.distributed.destroy()


def _train(
    settings: _settings.Settings,
    logger: logging.Logger,
    training_stats: stats.Stats,
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
        utils.distributed.get_rank(settings.distributed),
    )

    # build data
    train_dataset = _create_dataset(settings, logger=logger)

    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if settings.distributed.use_nccl
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=False,
        batch_size=settings.model.batch_size,
        num_workers=settings.data.workers,
        pin_memory=True,
        drop_last=settings.model.drop_last,
        worker_init_fn=utils.distributed.set_dataloader_seeds,
    )

    logger.info(
        "Building data done, dataset size: %s samples", len(train_dataset)
    )
    logger.info("Batches per epoch: %s", len(train_loader))

    device = utils.distributed.get_device(settings.distributed)

    # build model
    model = models.resnet_dcv2.Models[settings.model.architecture](
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
        logger.info(
            "Number of trainable parameters: %s",
            utils.models.get_number_of_trainable_parameters(model),
        )
        logger.info(
            "Number of non-trainable parameters: %s",
            utils.models.get_number_of_non_trainable_parameters(model),
        )

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
    logger.info("Iters %s", iters)
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
            output_device=settings.distributed.local_rank,
        )

    restored_variables = models.checkpoints.restart_from_checkpoint(
        path=settings.dump.checkpoints,
        model=model,
        optimizer=optimizer,
        properties=settings.distributed,
    )
    start_epoch = restored_variables["epoch"]

    # build the memory bank
    mb_path = settings.dump.path / f"mb-{settings.distributed.global_rank}.pth"

    if mb_path.is_file():
        mb_ckp = torch.load(mb_path)
        local_memory_index = mb_ckp["local_memory_index"]
        local_memory_embeddings = mb_ckp["local_memory_embeddings"]
    else:
        local_memory_index, local_memory_embeddings = cluster.init_embeddings(
            dataloader=train_loader,
            model=model,
            settings=settings,
            device=device,
        )

    cudnn.benchmark = True

    train_time_start = time.time()
    timer = _timer.CPUGPUTimer()

    for epoch in range(start_epoch, settings.model.epochs):
        epoch_start_time = time.time()
        timer.start(_timer.TimeType.EPOCH)

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
            timer=timer,
        )
        training_stats.update(scores)

        # save checkpoints
        if utils.distributed.is_primary_device():
            epoch_time = time.time() - epoch_start_time
            logger.info("Epoch time: %s s", epoch_time)

            if settings.enable_tracking:
                mantik.mlflow.log_metrics(
                    {"epoch_time_s": epoch_time},
                    step=epoch,
                )

            models.checkpoints.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=settings.dump.checkpoints,
                checkpoint_freq=settings.dump.checkpoint_freq,
                target_epochs=settings.model.epochs,
            )
        torch.save(
            {
                "local_memory_embeddings": local_memory_embeddings,
                "local_memory_index": local_memory_index,
            },
            mb_path,
        )
        if utils.distributed.is_primary_device():
            epoch_time = time.time() - epoch_start_time
            logger.info("Epoch time: %s s", epoch_time)

            if settings.enable_tracking:
                mantik.mlflow.log_metrics(
                    {"epoch_time_s": epoch_time},
                    step=epoch,
                )
        timer.end(_timer.TimeType.EPOCH)

    train_time = time.time() - train_time_start

    if utils.distributed.is_primary_device() and settings.enable_tracking:
        mantik.mlflow.log_metric("train_time_s", train_time)

        timer.log_mlflow_all("deep500")
        stream = io.StringIO()
        timer.print_all_time_stats(stream)
        stream.seek(0)
        mantik.mlflow.log_text(stream.read(), "deep500-results.txt")

    logger.info("Total training time: %s s", train_time)


def _create_dataset(
    settings: _settings.Settings, logger: logging.Logger
) -> datasets.crop.Base:
    if settings.data.use_mnist:
        return datasets.crop.MultiCropMnistDataset(
            data_path=settings.data.path,
            nmb_crops=settings.preprocessing.nmb_crops,
            size_crops=settings.preprocessing.size_crops,
            min_scale_crops=settings.preprocessing.min_scale_crops,
            max_scale_crops=settings.preprocessing.max_scale_crops,
            return_index=True,
        )
    elif settings.data.use_imagenet:
        return datasets.crop.MultiCropImageNet(
            data_path=settings.data.path,
            nmb_crops=settings.preprocessing.nmb_crops,
            size_crops=settings.preprocessing.size_crops,
            min_scale_crops=settings.preprocessing.min_scale_crops,
            max_scale_crops=settings.preprocessing.max_scale_crops,
            return_index=True,
        )

    if settings.data.pattern is not None:
        # If a data pattern is given, it is assumed that the
        # given data path is a folder with netCDF files.
        logger.warning("Assuming xarray.Dataset from netCDF files")
        ds = datasets.dwd.get_dwd_era5_data(
            path=settings.data.path,
            pattern=settings.data.pattern,
            levels=settings.data.levels,
            parallel_loading=settings.data.parallel_loading,
            select_dwd_area=settings.data.select_dwd_area,
        )
        return datasets.crop.MultiCropXarrayDataset(
            data_path=settings.data.path,
            dataset=ds,
            nmb_crops=settings.preprocessing.nmb_crops,
            size_crops=settings.preprocessing.size_crops,
            min_scale_crops=settings.preprocessing.min_scale_crops,
            max_scale_crops=settings.preprocessing.max_scale_crops,
            return_index=True,
        )
    logger.warning("Assuming image dataset")
    return datasets.crop.MultiCropDataset(
        data_path=settings.data.path,
        nmb_crops=settings.preprocessing.nmb_crops,
        size_crops=settings.preprocessing.size_crops,
        min_scale_crops=settings.preprocessing.min_scale_crops,
        max_scale_crops=settings.preprocessing.max_scale_crops,
        return_index=True,
    )


def _log_stdout_stderr(stdout: str | None, stderr: str | None) -> None:
    if stdout is not None:
        mantik.mlflow.log_artifact(stdout)
    if stderr is not None:
        mantik.mlflow.log_artifact(stderr)
