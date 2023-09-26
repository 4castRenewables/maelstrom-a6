# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
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

import a6.dcv2._checkpoints as _checkpoints
import a6.dcv2._initialization as _initialization
import a6.dcv2._parse as _parse
import a6.dcv2.cluster as cluster
import a6.dcv2.dataset as dataset
import a6.dcv2.logs as logs
import a6.dcv2.models as models
import a6.dcv2.train as train
import a6.utils as utils
import a6.utils.mantik as mantik
import mlflow


@errors.record
def train_dcv2(raw_args: list[str] | None = None):
    args = _parse.create_argparser().parse_args(raw_args)

    args.node_id = utils.slurm.get_node_id()
    utils.distributed.set_required_env_vars(args)

    logger, training_stats = _initialization.initialize_logging(
        args, columns=("epoch", "loss")
    )

    # Args of type list have to be manually set since MLflow doesn't allow
    # to pass lists.
    # ``crops_for_assign`` is the indexes of the crops used for clustering.
    _parse.overwrite_arg(
        args,
        attribute="crops_for_assign",
        value=[i for i in range(args.nmb_crops[0])],
    )
    # ``nmb_prototypes`` is a list with the number of clusters (``K``) to use
    # for the K-means. The length of the list (``nmb_clusters``) defines how
    # many times the clustering is performed.
    _parse.overwrite_arg(
        args,
        attribute="nmb_prototypes",
        value=[args.nmb_clusters for _ in range(args.nmb_prototypes)],
    )

    logger.info(
        (
            "Required env vars for distributed mode set: "
            "RANK=%s, LOCAL_RANK=%s, WORLD_SIZE=%s"
        ),
        args.global_rank,
        args.local_rank,
        args.world_size,
    )

    utils.distributed.setup(args)

    if utils.distributed.is_primary_device():
        start = time.time()

        utils.logging.log_env_vars()

        stdout = utils.slurm.get_stdout_file()
        stderr = utils.slurm.get_stderr_file()

        if args.enable_tracking:
            mantik.call_mlflow_method(mlflow.start_run)

            if utils.slurm.is_slurm_job():
                mantik.call_mlflow_method(
                    mlflow.log_params,
                    utils.slurm.get_slurm_env_vars(),
                )

    _train(
        args=args,
        logger=logger,
        training_stats=training_stats,
    )

    if utils.distributed.is_primary_device():
        logging.info("All done!")
        runtime = time.time() - start
        logging.info("Total runtime (s): %s", runtime)

        if args.enable_tracking:
            mantik.call_mlflow_method(
                mlflow.log_metric, "total_runtime_s", runtime
            )

            _log_stdout_stderr(stdout=stdout, stderr=stderr)

            # The following files are saved to disk in `a6.dcv2.cluster.py`
            mantik.call_mlflow_method(mlflow.log_artifacts, args.dump_results)

            mantik.call_mlflow_method(mlflow.end_run)

    utils.distributed.destroy()


def _train(
    args,
    logger: logging.Logger,
    training_stats: logs.Stats,
):
    logger.info(
        (
            "Starting training on host %s (node %s), global rank %s, "
            "local rank %s (torch.distributed.get_rank()=%s)"
        ),
        socket.gethostname(),
        args.node_id,
        args.global_rank,
        args.local_rank,
        torch.distributed.get_rank(),
    )

    # build data
    train_dataset = dataset.MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        return_index=True,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        # ``drop_last=True`` gives each device the same amount of samples,
        # but removes some from the clustering.
        drop_last=True,
    )
    logger.info("Building data done with %s images loaded", len(train_dataset))

    device = utils.distributed.get_device(args)

    # build model
    model = models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
        device=device,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        if args.use_cpu:
            pass  # model = nn.BatchNorm2d(2)(model)
        else:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        if utils.distributed.is_multi_gpu():
            import apex
            from apex.parallel.LARC import LARC

            # with apex syncbn we sync bn per group because it speeds up
            # computation compared to global syncbn
            process_group = apex.parallel.create_syncbn_process_group(
                args.syncbn_process_group_size
            )
            model = apex.parallel.convert_syncbn_model(
                model, process_group=process_group
            )
        else:
            logger.warning(
                "Sync batch norm 'apex' defined, but training is performed "
                "on single GPU, hence using native PyTorch"
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
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )

    if args.sync_bn == "apex":
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
        args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs
    )
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array(
        [
            args.final_lr
            + 0.5
            * (args.base_lr - args.final_lr)
            * (
                1
                + math.cos(
                    math.pi
                    * t
                    / (len(train_loader) * (args.epochs - args.warmup_epochs))
                )
            )
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done")

    # wrap model
    if not args.use_cpu:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True,
        )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    _checkpoints.restart_from_checkpoint(
        args.dump_path / "checkpoint.pth.tar",
        args=args,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    # build the memory bank
    mb_path = args.dump_path / f"mb-{args.global_rank}.pth"

    if mb_path.is_file():
        mb_ckp = torch.load(mb_path)
        local_memory_index = mb_ckp["local_memory_index"]
        local_memory_embeddings = mb_ckp["local_memory_embeddings"]
    else:
        local_memory_index, local_memory_embeddings = cluster.init_memory(
            dataloader=train_loader, model=model, args=args, device=device
        )

    cudnn.benchmark = True
    for epoch in range(start_epoch, args.epochs):
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
            args=args,
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
                args.dump_path / "checkpoint.pth.tar",
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    args.dump_path / "checkpoint.pth.tar",
                    args.dump_checkpoints / f"checkpoint-epoch-{epoch}.pth",
                )
        torch.save(
            {
                "local_memory_embeddings": local_memory_embeddings,
                "local_memory_index": local_memory_index,
            },
            mb_path,
        )


def _log_stdout_stderr(stdout: str, stderr: str | None) -> None:
    mantik.call_mlflow_method(mlflow.log_artifact, stdout)
    if stderr is not None:
        mantik.call_mlflow_method(mlflow.log_artifact, stderr)
