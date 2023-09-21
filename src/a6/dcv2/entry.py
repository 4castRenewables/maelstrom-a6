# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import math
import os
import shutil

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import a6.dcv2._checkpoints as _checkpoints
import a6.dcv2._initialization as _initialization
import a6.dcv2._parse as _parse
import a6.dcv2.cluster as cluster
import a6.dcv2.dataset as dataset
import a6.dcv2.models as models
import a6.dcv2.train as train
import a6.utils as utils
import a6.utils.mantik as mantik
import mlflow


def train_dcv2():
    args = _parse.create_argparser().parse_args()
    utils.distributed.init_distributed_mode(args)
    _initialization.fix_random_seeds(args.seed)
    logger, training_stats = _initialization.initialize_exp(
        args, "epoch", "loss"
    )

    utils.logging.log_env_vars()

    if args.enable_tracking and utils.distributed.is_primary_device():
        slurm_job_id = utils.slurm.get_slurm_job_id()

        if slurm_job_id is not None:
            mantik.call_mlflow_method(
                mlflow.start_run,
                run_name=f"slurm-{slurm_job_id}-node-{utils.slurm.get_node_id()}",  # noqa: E501
            )

            mantik.call_mlflow_method(
                mlflow.log_params,
                utils.slurm.get_slurm_env_vars(),
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
        import apex
        from apex.parallel.LARC import LARC

        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(
            args.syncbn_process_group_size
        )
        model = apex.parallel.convert_syncbn_model(
            model, process_group=process_group
        )

    if not args.use_cpu:
        # copy model to GPU
        model = model.cuda()

    if args.rank == 0:
        logger.info(model)

    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )

    if args.sync_bn == "apex":
        optimizer = LARC(
            optimizer=optimizer, trust_coefficient=0.001, clip=False
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
    logger.info("Building optimizer done.")

    # wrap model
    if not args.use_cpu:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu_to_work_on],
            find_unused_parameters=True,
        )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    _checkpoints.restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        args=args,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    # build the memory bank
    mb_path = os.path.join(args.dump_path, "mb" + str(args.rank) + ".pth")
    if os.path.isfile(mb_path):
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
        logger.info("============ Starting epoch %i ... ============", epoch)

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
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(
                        args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"
                    ),
                )
        torch.save(
            {
                "local_memory_embeddings": local_memory_embeddings,
                "local_memory_index": local_memory_index,
            },
            mb_path,
        )
