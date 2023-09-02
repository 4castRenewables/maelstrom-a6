# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import logging
import time

import torch.nn as nn

import a6.dcv2.cluster as cluster
import a6.dcv2.utils as utils

logger = logging.getLogger(__name__)


def train(
    dataloader,
    model,
    optimizer,
    epoch,
    schedule,
    local_memory_index,
    local_memory_embeddings,
    args,
    device,
):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    assignments = cluster.cluster_memory(
        model=model,
        local_memory_index=local_memory_index,
        local_memory_embeddings=local_memory_embeddings,
        size_dataset=len(dataloader.dataset),
        args=args,
        device=device,
    )
    logger.info("Clustering for epoch %i done", epoch)

    end = time.time()
    start_idx = 0
    for it, (idx, inputs) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(dataloader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule[iteration]

        # ============ multi-res forward passes ... ============
        emb, output = model(inputs)
        emb = emb.detach()
        bs = inputs[0].size(0)

        # ============ deepcluster-v2 loss ... ============
        loss = 0
        for h in range(len(args.nmb_prototypes)):
            scores = output[h] / args.temperature
            targets = (
                assignments[h][idx]
                .repeat(sum(args.nmb_crops))
                .to(device=device, non_blocking=True)
            )
            loss += cross_entropy(scores, targets)
        loss /= len(args.nmb_prototypes)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        loss.backward()
        # cancel some gradients
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ update memory banks ... ============
        local_memory_index[start_idx : start_idx + bs] = idx  # noqa: E203
        for i, crop_idx in enumerate(args.crops_for_assign):
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
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: %i[{1}]\t"
                "Iteration: %i\t"
                "Time: %s (%s)\t"
                "Data: %s (%s)\t"
                "Loss: %s (%s)\t"
                "Lr: %s",
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
    return (epoch, losses.avg), local_memory_index, local_memory_embeddings
