# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import contextlib
import logging
import math
import os
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
import a6.dcv2._logs as _logs
import a6.dcv2._parse as _parse
import a6.dcv2._settings as _settings
import a6.dcv2.cluster as cluster
import a6.dcv2.train as train
import a6.features as features
import a6.models as models
import a6.utils as utils
import a6.utils.mantik as mantik
import mlflow


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

    with setup_distributed(
        settings=settings, logger=logger
    ), energy_profiler() as measured_scope:
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
        settings, columns=("epoch", "loss")
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
            mantik.call_mlflow_method(mlflow.start_run)

            if utils.slurm.is_slurm_job():
                mantik.call_mlflow_method(
                    mlflow.log_params,
                    utils.slurm.get_slurm_env_vars(),
                )

    yield

    if utils.distributed.is_primary_device():
        logger.info("All done!")
        runtime = time.time() - start
        logger.info("Total runtime: %s s", runtime)

        if settings.enable_tracking:
            mantik.call_mlflow_method(
                mlflow.log_metric, "total_runtime_s", runtime
            )

            _log_stdout_stderr(stdout=stdout, stderr=stderr)

            # The following files are saved to disk in `a6.dcv2.cluster.py`
            mantik.call_mlflow_method(
                # NOTE: Only log plots due to large size of other files
                mlflow.log_artifacts,
                settings.dump.plots,
            )

            mantik.call_mlflow_method(mlflow.end_run)

    utils.distributed.destroy()


def _train(
    settings: _settings.Settings,
    logger: logging.Logger,
    training_stats: _logs.Stats,
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

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
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
    model = models.resnet.Models[settings.model.architecture](
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
        local_memory_index, local_memory_embeddings = cluster.init_embeddings(
            dataloader=train_loader,
            model=model,
            settings=settings,
            device=device,
        )

    cudnn.benchmark = True

    train_time_start = time.time()

    for epoch in range(start_epoch, settings.model.epochs):
        start = time.time()

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
            epoch_time = time.time() - start
            logger.info("Epoch time: %s s", epoch_time)

            if settings.enable_tracking:
                utils.mantik.call_mlflow_method(
                    mlflow.log_metrics,
                    {"epoch_time_s": epoch_time},
                    step=epoch,
                )

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

    logger.info("Total training time: %s s", time.time() - train_time_start)


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

    if settings.data.pattern is not None:
        # If a data pattern is given, it is assumed that the
        # given data path is a folder with netCDF files.
        logger.warning("Assuming xarray.Dataset from netCDF files")
        coordinates = datasets.coordinates.Coordinates()
        variables = datasets.variables.Model()
        drop_variables = settings.data.drop_variables or []

        preprocessing = (
            (
                datasets.methods.select.select_dwd_area(coordinates=coordinates)
                if settings.data.select_dwd_area
                else datasets.methods.identity.identity()
            )
            >> features.methods.weighting.weight_by_latitudes(
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

        ds = datasets.Era5(
            path=settings.data.path,
            pattern=settings.data.pattern,
            preprocessing=preprocessing,
            parallel_loading=settings.data.parallel_loading,
        ).to_xarray(levels=settings.data.levels)
        logger.info(
            "Reading data from netCDF files and converting to xarray.Dataset"
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
        mantik.call_mlflow_method(mlflow.log_artifact, stdout)
    if stderr is not None:
        mantik.call_mlflow_method(mlflow.log_artifact, stderr)
