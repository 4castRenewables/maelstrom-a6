# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import logging
import pathlib

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from scipy.sparse import csr_matrix

import a6.plotting as plotting
import a6.utils as utils
import a6.utils.mantik as mantik
import mlflow

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
    epoch: int,
    model,
    local_memory_index: torch.Tensor,
    local_memory_embeddings: torch.Tensor,
    size_dataset: int,
    args,
    device: torch.device,
    nmb_kmeans_iters=10,
):
    # j defines which crops are used for the K-means run.
    # E.g. if the number of crops (``self.nmb_mbs``) is 2, and
    # ``self.num_clusters = [30, 30, 30, 30]``, the crops will
    # be used as following:
    #
    # 1. K=30, j=0
    # 2. K=30, j=1
    # 3. K=30, j=0
    # 4. K=30, j=1
    j = 0

    assignments = (
        -100 * torch.ones(len(args.nmb_prototypes), size_dataset).long()
    )
    indexes = -100 * torch.ones(len(args.nmb_prototypes), size_dataset).long()
    embeddings = -100 * torch.ones(
        len(args.nmb_prototypes),
        len(args.crops_for_assign),
        size_dataset,
        args.feat_dim,
    )
    distances = -100 * torch.ones(len(args.nmb_prototypes), size_dataset)

    if args.use_cpu:
        embeddings = embeddings.float()
        distances = distances.float()
    else:
        embeddings = embeddings.half()
        distances = distances.half()

    with torch.no_grad():
        for i_K, K in enumerate(args.nmb_prototypes):
            # run distributed k-means

            # init centroids with elements from memory bank of rank 0
            centroids = torch.empty(K, args.feat_dim).to(
                device=device, non_blocking=True
            )
            if args.rank == 0:
                # Init centroids with elements from memory bank of rank 0 by
                # taking K random samples from its local memory embeddings
                # (i.e. the cropped samples) as centroids
                random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
                assert (
                    len(random_idx) >= K
                ), "please reduce the number of centroids"
                centroids = local_memory_embeddings[j][random_idx]

            # Send random centroids from rank 0 to all processes
            dist.broadcast(centroids, 0)

            for n_iter in range(nmb_kmeans_iters + 1):
                # E step
                dot_products = torch.mm(
                    local_memory_embeddings[j], centroids.t()
                )
                local_distances, local_assignments = dot_products.max(dim=1)

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

            getattr(
                model.prototypes if args.use_cpu else model.module.prototypes,
                "prototypes" + str(i_K),
            ).weight.copy_(centroids)
            assignments_all = utils.distributed.gather_from_all_ranks(
                local_assignments
            )
            indexes_all = utils.distributed.gather_from_all_ranks(
                local_memory_index
            )
            # To gather embeddings, make sure to gather using
            # ``local_memory_embeddings[j]``
            embeddings_all = utils.distributed.gather_from_all_ranks(
                local_memory_embeddings[j]
            )
            distances_all = utils.distributed.gather_from_all_ranks(
                local_distances
            )

            # log assignments
            assignments[i_K][indexes_all] = assignments_all
            logger.info("Assigments: %s", assignments_all)

            assignments[i_K] = -1
            assignments[i_K][indexes_all] = assignments_all
            indexes[i_K] = -1
            indexes[i_K][indexes_all] = indexes_all
            distances[i_K] = -1.0
            distances[i_K][indexes_all] = distances_all
            # For the embeddings, make sure to use j for indexing
            embeddings[i_K][j] = -1.0
            embeddings[i_K][j][indexes_all] = embeddings_all

            j_prev = j
            # next memory bank to use
            j = (j + 1) % len(args.crops_for_assign)

        epoch_comp = epoch + 1

        if (
            epoch_comp == 1
            or (epoch_comp <= 100 and epoch_comp % 25 == 0)
            or epoch_comp % 100 == 0
        ):
            logging.info(
                "Saving clustering data on rank %s at epoch %s",
                utils.distributed.get_rank(),
                epoch,
            )

            torch.save(
                centroids,
                _create_path(
                    path=args.dump_path, file_name="centroids.pt", epoch=epoch
                ),
            )
            torch.save(
                assignments,
                _create_path(
                    path=args.dump_path, file_name="assignments.pt", epoch=epoch
                ),
            )
            torch.save(
                embeddings,
                _create_path(
                    path=args.dump_path, file_name="embeddings.pt", epoch=epoch
                ),
            )
            torch.save(
                indexes,
                _create_path(
                    path=args.dump_path, file_name="indexes.pt", epoch=epoch
                ),
            )
            torch.save(
                distances,
                _create_path(
                    path=args.dump_path, file_name="distances.pt", epoch=epoch
                ),
            )

            if utils.distributed.is_primary_device():
                # Save which random samples were used as the centroids.
                torch.save(
                    random_idx,
                    _create_path(
                        path=args.dump_path,
                        file_name="centroid-indexes.pt",
                        epoch=epoch,
                    ),
                )
                plots_dir = args.dump_path / "plots"
                plotting.embeddings.plot_embeddings_using_tsne(
                    embeddings=embeddings[-1],
                    # Use previous j since this represents which crops
                    # were used for last cluster iteration.
                    j=j_prev,
                    assignments=assignments[-1],
                    centroids=random_idx,
                    name=f"epoch-{epoch}-embeddings",
                    output_dir=plots_dir,
                )
                plotting.assignments.plot_abundance(
                    assignments=assignments[-1],
                    name=f"epoch-{epoch}-assignments-abundance",
                    output_dir=plots_dir,
                )
                plotting.assignments.plot_appearance_per_week(
                    assignments=assignments[-1],
                    name=f"epoch-{epoch}-appearance-per-week",
                    output_dir=plots_dir,
                )

                n_unassigned_samples = _calculate_number_of_unassigned_samples(
                    assignments[-1],
                )
                percent_unassigned_samples = (
                    n_unassigned_samples / assignments.shape[-1]
                )
                logger.warning(
                    "Number of unassigned samples: %s (%s%)",
                    n_unassigned_samples,
                    percent_unassigned_samples,
                )

                if args.enable_tracking:
                    mantik.call_mlflow_method(
                        mlflow.log_metric,
                        "unassigned_samples",
                        n_unassigned_samples,
                    )
                    mantik.call_mlflow_method(
                        mlflow.log_metric,
                        "unassigned_samples_percent",
                        percent_unassigned_samples,
                    )

    return assignments


def _get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix(
        (cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size)
    )
    return [np.unravel_index(row.data, data.shape) for row in M]


def _create_path(path: pathlib.Path, file_name: str, epoch: int) -> str:
    return f"{path}/epoch-{epoch}-{file_name}"


def _calculate_number_of_unassigned_samples(assignments: torch.Tensor) -> int:
    unassigned_sample_indexes = (assignments == -1).nonzero(as_tuple=True)[0]
    n_unassigned_samples = len(unassigned_sample_indexes)
    return n_unassigned_samples
