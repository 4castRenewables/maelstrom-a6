# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import logging
import time

import torch.nn as nn
import torch.utils.data

import a6.dcv2._averaging as _averaging
import a6.dcv2._settings as _settings
import a6.dcv2.cluster as cluster
import a6.utils as utils
import mlflow

logger = logging.getLogger(__name__)


def train(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    schedule,
    local_memory_index: torch.Tensor,
    local_memory_embeddings: torch.Tensor,
    settings: _settings.Settings,
    device: torch.device,
):
    batch_time = _averaging.AverageMeter()
    data_time = _averaging.AverageMeter()
    losses = _averaging.AverageMeter()
    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=cluster.IGNORE_INDEX)

    assignments = cluster.cluster_memory(
        epoch=epoch,
        model=model,
        local_memory_index=local_memory_index,
        local_memory_embeddings=local_memory_embeddings,
        size_dataset=len(dataloader.dataset),
        settings=settings,
        device=device,
    )

    logger.info("Clustering for epoch %i done", epoch)

    end = time.time()
    start_idx = 0
    for it, (idx, inputs) in enumerate(dataloader):
        logger.debug("Calculating loss for index %s", idx)

        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(dataloader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule[iteration]

        # ============ multi-res forward passes ... ============
        # Output here returns the output for each head (prototype)
        # and hence has size ``len(settings.model.nmb_prototypes)``.
        emb, output = model(inputs)
        emb = emb.detach()
        bs = inputs[0].size(0)

        logger.debug("Batch size is %s", bs)

        # ============ deepcluster-v2 loss ... ============
        loss = 0
        for h in range(len(settings.model.nmb_prototypes)):
            scores = output[h] / settings.model.temperature
            targets = (
                assignments[h][idx]
                .repeat(sum(settings.preprocessing.nmb_crops))
                .to(device=device, non_blocking=True)
            )
            loss += cross_entropy(scores, targets)
        loss /= len(settings.model.nmb_prototypes)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        loss.backward()
        # cancel some gradients
        if iteration < settings.model.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ update memory banks ... ============
        local_memory_index[start_idx : start_idx + bs] = idx  # noqa: E203
        for i, crop_idx in enumerate(settings.preprocessing.crops_for_assign):
            local_memory_embeddings[i][
                start_idx : start_idx + bs  # noqa: E203
            ] = emb[
                crop_idx * bs : (crop_idx + 1) * bs  # noqa: E203
            ]
        start_idx += bs

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if utils.distributed.is_primary_device() and it % 50 == 0:
            logger.info(
                "[EPOCH %i, ITERATION %i] "
                "batch time: %s (%s) "
                "data load time: %s (%s) "
                "loss: %s (%s) "
                "lr: %s",
                epoch,
                it,
                batch_time.val,
                batch_time.avg,
                data_time.val,
                data_time.avg,
                losses.val,
                losses.avg,
                optimizer.state_dict()["param_groups"][0]["lr"],
            )

    if settings.enable_tracking and utils.distributed.is_primary_device():
        metrics = {
            "batch_time": batch_time.val,
            "batch_time_avg": batch_time.avg,
            "data_load_time": data_time.val,
            "data_load_time_avg": data_time.avg,
            "loss": losses.val,
            "loss_avg": losses.avg,
        }

        utils.mantik.call_mlflow_method(
            mlflow.log_metrics,
            metrics,
            step=epoch,
        )

    if utils.distributed.is_primary_device() and (device.type == "cuda"):
        logging.info(
            "========= Memory Summary at epoch %s =======\n%s\n",
            epoch,
            torch.cuda.memory_summary(),
        )

    return (epoch, losses.avg), local_memory_index, local_memory_embeddings
