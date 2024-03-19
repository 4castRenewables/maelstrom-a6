# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import logging
import time

import mantik.mlflow
import numpy as np
import torch.nn as nn
import torch.utils.data

import a6.dcv2.averaging as _averaging
import a6.dcv2.cluster as cluster
import a6.dcv2.settings as _settings
import a6.utils as utils

_timer = utils.benchmark.import_deep500()

logger = logging.getLogger(__name__)


def train(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    schedule: np.ndarray,
    local_memory_index: torch.Tensor,
    local_memory_embeddings: torch.Tensor,
    settings: _settings.Settings,
    device: torch.device,
    timer,
):
    batch_time = _averaging.AverageMeter()
    data_time = _averaging.AverageMeter()
    forward_time = _averaging.AverageMeter()
    loss_time = _averaging.AverageMeter()
    backward_time = _averaging.AverageMeter()
    embeddings_time = _averaging.AverageMeter()
    losses = _averaging.AverageMeter()

    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=cluster.IGNORE_INDEX)

    assignments = cluster.cluster_embeddings(
        epoch=epoch,
        model=model,
        local_memory_index=local_memory_index,
        local_memory_embeddings=local_memory_embeddings,
        size_dataset=len(dataloader.dataset),
        settings=settings,
        device=device,
    )

    end = time.time()
    start_idx = 0

    timer.start(_timer.TimeType.BATCH, gpu=True)
    timer.start(_timer.TimeType.IO)

    for it, (inputs, idx) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        timer.end(_timer.TimeType.IO)

        logger.debug("Calculating loss for index %s", idx)

        for crop_index, inp in enumerate(inputs):
            for index in range(inp.size(0)):
                sample = inp[index]
                if torch.isnan(sample).any():
                    logger.warning(
                        (
                            "Input at crop index %i and index %i has NaN "
                            "values: %s"
                        ),
                        crop_index,
                        idx[index],
                        sample,
                    )

        # update learning rate
        iteration = epoch * len(dataloader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule[iteration]

        # ============ multi-res forward passes ... ============
        # Output here returns the output for each head (prototype)
        # and hence has size ``len(settings.model.nmb_prototypes)``.
        start_forward = time.time()
        timer.start(_timer.TimeType.FORWARD, gpu=True)

        emb, output = model(inputs)
        emb = emb.detach()
        bs = inputs[0].size(0)

        forward_time.update(time.time() - start_forward)
        timer.end(_timer.TimeType.FORWARD, gpu=True)

        if bs == 0:
            raise RuntimeError(
                f"Batch size is zero, loss will be NaN: it={it}, idx={idx}, "
                "inputs[0]={inputs[0]}"
            )

        logger.debug("Batch size for iteration %i is %s (idx)", it, bs)

        # ============ deepcluster-v2 loss ... ============
        start_loss = time.time()
        timer.start(_timer.TimeType.OTHER, gpu=True)

        loss = torch.tensor(0.0).to(device=device)
        for h in range(len(settings.model.nmb_prototypes)):
            scores = output[h] / settings.model.temperature
            targets = (
                assignments[h][idx]
                .repeat(sum(settings.preprocessing.nmb_crops))
                .to(device=device, non_blocking=True)
            )
            loss_temp = cross_entropy(scores, targets)
            loss += loss_temp

            if not settings.testing and (
                torch.isnan(loss_temp).any() or torch.isnan(loss).any()
            ):
                logger.exception(
                    (
                        "Loss is NaN: it=%i, prototype(h)=%i, "
                        "nmb_prototypes=%s, inputs_is_any_nan=%s"
                        "idx=%s, assignments=%s, output=%s, targets=%s, "
                        "scores=%s, sum_nmb_crops=%s, loss_temp=%s, loss=%s"
                    ),
                    it,
                    h,
                    settings.model.nmb_prototypes,
                    any(torch.isnan(inp).any() for inp in inputs),
                    idx,
                    assignments[h][idx],
                    output[h],
                    targets,
                    scores,
                    sum(settings.preprocessing.nmb_crops),
                    loss_temp.item(),
                    loss.item(),
                )

                raise RuntimeError("Loss exploded to NaN")

        loss /= len(settings.model.nmb_prototypes)

        loss_time.update(time.time() - start_loss)
        timer.end(_timer.TimeType.OTHER, gpu=True)

        # ============ backward and optim step ... ============
        start_backward = time.time()
        timer.start(_timer.TimeType.BACKWARD, gpu=True)

        optimizer.zero_grad()
        loss.backward()
        # cancel some gradients
        if iteration < settings.model.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        backward_time.update(time.time() - start_backward)
        timer.end(_timer.TimeType.BACKWARD, gpu=True)

        # ============ update memory banks ... ============
        start_embeddings = time.time()

        local_memory_index[start_idx : start_idx + bs] = idx  # noqa: E203
        for i, crop_idx in enumerate(settings.model.crops_for_assign):
            local_memory_embeddings[i][
                start_idx : start_idx + bs  # noqa: E203
            ] = emb[
                crop_idx * bs : (crop_idx + 1) * bs  # noqa: E203
            ]
        start_idx += bs

        embeddings_time.update(time.time() - start_embeddings)

        # ============ misc ... ============
        losses.update(loss.item(), bs)
        batch_time.update(time.time() - end)
        timer.end(_timer.TimeType.BATCH, gpu=True)
        end = time.time()

        if it % 10 == 0:
            # Prevent CUDA event exhaustion.
            timer.complete_all()

        if it != len(dataloader) - 1:
            timer.start(_timer.TimeType.BATCH, gpu=True)
            timer.start(_timer.TimeType.IO)

        log_metrics = True if settings.verbose else it % 50 == 0
        if utils.distributed.is_primary_device() and log_metrics:
            logger.info(
                "[EPOCH %i, ITERATION %i] "
                "batch time: %s s, "
                "batch time avg: %s s, "
                "data load time: %s s, "
                "data load time avg: %s s, "
                "forward time: %s s, "
                "forward time avg: %s s, "
                "loss time: %s s, "
                "loss time avg: %s s, "
                "backward time: %s s, "
                "backward time avg: %s s, "
                "embeddings time: %s s, "
                "embeddings time avg: %s s, "
                "loss: %s, "
                "loss avg: %s, "
                "lr: %s",
                epoch,
                it,
                batch_time.val,
                batch_time.avg,
                data_time.val,
                data_time.avg,
                forward_time.val,
                forward_time.avg,
                loss_time.val,
                loss_time.avg,
                backward_time.val,
                backward_time.avg,
                embeddings_time.val,
                embeddings_time.avg,
                losses.val,
                losses.avg,
                optimizer.state_dict()["param_groups"][0]["lr"],
            )

    if settings.enable_tracking and utils.distributed.is_primary_device():
        metrics = {
            "batch_time_avg_s": batch_time.avg,
            "data_load_time_avg_s": data_time.avg,
            "forward_time_avg_s": forward_time.avg,
            "loss_time_avg_s": loss_time.avg,
            "backward_time_avg_s": backward_time.avg,
            "embeddings_time_avg_s": embeddings_time.avg,
            "loss_avg": losses.avg,
        }

        mantik.mlflow.log_metrics(
            metrics,
            step=epoch,
        )

    if utils.distributed.is_primary_device():
        utils.usage.log_cpu_memory_usage()

        if not settings.distributed.use_cpu:
            utils.usage.log_gpu_memory_usage(device)
            logger.info(
                "========= Memory Summary at epoch %s =======\n%s\n",
                epoch,
                torch.cuda.memory_summary(),
            )

    return (epoch, losses.avg), local_memory_index, local_memory_embeddings
