# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import logging

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def init_memory(dataloader, model, args, device):
    size_memory_per_process = len(dataloader) * args.batch_size
    local_memory_index = (
        torch.zeros(size_memory_per_process).long().to(device=device)
    )
    local_memory_embeddings = torch.zeros(
        len(args.crops_for_assign), size_memory_per_process, args.feat_dim
    ).to(device=device)
    start_idx = 0
    with torch.no_grad():
        logger.info("Start initializing the memory banks")
        for index, inputs in dataloader:
            nmb_unique_idx = inputs[0].size(0)
            index = index.to(device=device, non_blocking=True)

            # get embeddings
            outputs = []
            for crop_idx in args.crops_for_assign:
                inp = inputs[crop_idx].to(device=device, non_blocking=True)
                outputs.append(model(inp)[0])

            # fill the memory bank
            local_memory_index[
                start_idx : start_idx + nmb_unique_idx  # noqa: E203
            ] = index
            for mb_idx, embeddings in enumerate(outputs):
                local_memory_embeddings[mb_idx][
                    start_idx : start_idx + nmb_unique_idx  # noqa: E203
                ] = embeddings
            start_idx += nmb_unique_idx
    logger.info("Initialization of the memory banks done.")
    return local_memory_index, local_memory_embeddings


def cluster_memory(
    model,
    local_memory_index,
    local_memory_embeddings,
    size_dataset,
    args,
    device,
    nmb_kmeans_iters=10,
):
    j = 0
    assignments = (
        -100 * torch.ones(len(args.nmb_prototypes), size_dataset).long()
    )
    with torch.no_grad():
        for i_K, K in enumerate(args.nmb_prototypes):
            # run distributed k-means

            # init centroids with elements from memory bank of rank 0
            centroids = torch.empty(K, args.feat_dim).to(
                device=device, non_blocking=True
            )
            if args.rank == 0:
                random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
                assert (
                    len(random_idx) >= K
                ), "please reduce the number of centroids"
                centroids = local_memory_embeddings[j][random_idx]
            dist.broadcast(centroids, 0)

            for n_iter in range(nmb_kmeans_iters + 1):
                # E step
                dot_products = torch.mm(
                    local_memory_embeddings[j], centroids.t()
                )
                _, local_assignments = dot_products.max(dim=1)

                # finish
                if n_iter == nmb_kmeans_iters:
                    break

                # M step
                where_helper = _get_indices_sparse(
                    local_assignments.cpu().numpy()
                )
                counts = (
                    torch.zeros(K).to(device=device, non_blocking=True).int()
                )
                emb_sums = torch.zeros(K, args.feat_dim).to(
                    device=device, non_blocking=True
                )
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = torch.sum(
                            local_memory_embeddings[j][where_helper[k][0]],
                            dim=0,
                        )
                        counts[k] = len(where_helper[k][0])
                dist.all_reduce(counts)
                mask = counts > 0
                dist.all_reduce(emb_sums)
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                # normalize centroids
                centroids = nn.functional.normalize(centroids, dim=1, p=2)

            getattr(model.prototypes if args.use_cpu else model.module.prototypes, "prototypes" + str(i_K)).weight.copy_(
                centroids
            )

            # gather the assignments
            assignments_all = torch.empty(
                args.world_size,
                local_assignments.size(0),
                dtype=local_assignments.dtype,
                device=local_assignments.device,
            )
            assignments_all = list(assignments_all.unbind(0))
            dist_process = dist.all_gather(
                assignments_all, local_assignments, async_op=True
            )
            dist_process.wait()
            assignments_all = torch.cat(assignments_all).cpu()

            # gather the indexes
            indexes_all = torch.empty(
                args.world_size,
                local_memory_index.size(0),
                dtype=local_memory_index.dtype,
                device=local_memory_index.device,
            )
            indexes_all = list(indexes_all.unbind(0))
            dist_process = dist.all_gather(
                indexes_all, local_memory_index, async_op=True
            )
            dist_process.wait()
            indexes_all = torch.cat(indexes_all).cpu()

            # log assignments
            assignments[i_K][indexes_all] = assignments_all
            logger.info(assignments_all)

            # next memory bank to use
            j = (j + 1) % len(args.crops_for_assign)

    return assignments


def _get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix(
        (cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size)
    )
    return [np.unravel_index(row.data, data.shape) for row in M]
