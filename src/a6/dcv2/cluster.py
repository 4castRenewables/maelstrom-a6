# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import logging
import math
import pathlib

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from scipy.sparse import csr_matrix

import a6.dcv2._settings as _settings
import a6.plotting as plotting
import a6.utils as utils
import a6.utils.mantik as mantik
import mlflow

logger = logging.getLogger(__name__)

IGNORE_INDEX = -1


def init_embeddings(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    settings: _settings.Settings,
    device,
):
    size_dataset = len(dataloader.dataset)
    logger.info("Dataset size is %s samples", size_dataset)

    size_memory_per_process = int(
        math.ceil(size_dataset / settings.distributed.world_size)
    )
    logger.info("Size memory per process is %s", size_memory_per_process)

    if settings.model.drop_last:
        size_memory_per_process -= (
            size_memory_per_process % settings.model.batch_size
        )
        logger.info(
            "Adjusted size of memory per process due to drop_last=True to %s",
            size_memory_per_process,
        )

    logger.info("Processing %s samples", size_memory_per_process)

    local_memory_index = (
        torch.zeros(size_memory_per_process).long().to(device=device)
    )
    local_memory_embeddings = torch.zeros(
        len(settings.model.crops_for_assign),
        size_memory_per_process,
        settings.model.feature_dimensions,
    ).to(device=device)
    start_idx = 0
    with torch.no_grad():
        logger.info("Start initializing the memory banks")
        for index, inputs in dataloader:
            nmb_unique_idx = inputs[0].size(0)
            index = index.to(device=device, non_blocking=True)

            # get embeddings
            outputs = []
            for crop_idx in settings.model.crops_for_assign:
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
    logger.debug(
        "Initialization of the memory banks done with %s local memory indexes",
        local_memory_index.size(),
    )
    return local_memory_index, local_memory_embeddings


def cluster_embeddings(
    epoch: int,
    model,
    local_memory_index: torch.Tensor,
    local_memory_embeddings: torch.Tensor,
    size_dataset: int,
    settings: _settings.Settings,
    device: torch.device,
    nmb_kmeans_iters=10,
):
    logger.debug("Clustering %s samples", size_dataset)

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

    n_heads = len(settings.model.nmb_prototypes)

    assignments = (IGNORE_INDEX * torch.ones(n_heads, size_dataset).long()).to(
        device
    )
    indexes = IGNORE_INDEX * torch.ones(n_heads, size_dataset).long().to(device)

    embeddings = float(IGNORE_INDEX) * torch.ones(
        n_heads,
        len(settings.model.crops_for_assign),
        size_dataset,
        settings.model.feature_dimensions,
    ).to(device)
    distances = float(IGNORE_INDEX) * torch.ones(n_heads, size_dataset).to(
        device
    )

    with torch.no_grad():
        for i_K, K in enumerate(settings.model.nmb_prototypes):
            # run distributed k-means

            # init centroids with elements from memory bank of rank 0
            centroids = torch.empty(K, settings.model.feature_dimensions).to(
                device=device, non_blocking=True
            )
            if settings.distributed.global_rank == 0:
                # Init centroids with elements from memory bank of rank 0 by
                # taking K random samples from its local memory embeddings
                # (i.e. the cropped samples) as centroids
                batch_size = len(local_memory_embeddings[j])
                random_idx = torch.randperm(batch_size)[:K]
                assert len(random_idx) >= K, (
                    f"Please reduce the number of centroids K={K}: "
                    f"K must be smaller than batch size {batch_size}"
                )
                centroids = local_memory_embeddings[j][random_idx]

            # Send random centroids from rank 0 to all processes
            dist.broadcast(centroids, src=0)

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
                emb_sums = torch.zeros(K, settings.model.feature_dimensions).to(
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

            # Copy centroids to model for forwarding
            getattr(
                model.prototypes
                if settings.distributed.use_cpu
                else model.module.prototypes,
                "prototypes" + str(i_K),
            ).weight.copy_(centroids)

            # Collect results
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

            logger.debug(
                "local_memory_embeddings[j]: %s embeddings_all: %s",
                local_memory_embeddings[j].size(),
                embeddings_all.size(),
            )

            logger.debug(
                (
                    "Results after gathering from all ranks: "
                    "embeddings: %s (size=%s), "
                    "assigments: %s (size=%s), "
                    "indexes: %s (size=%s)"
                ),
                embeddings_all,
                embeddings_all.size(),
                assignments_all,
                assignments_all.size(),
                indexes_all,
                indexes_all.size(),
            )

            for tensor, value in [
                (assignments_all, IGNORE_INDEX),
                (indexes_all, IGNORE_INDEX),
                (embeddings_all, float(IGNORE_INDEX)),
                (distances_all, float(IGNORE_INDEX)),
            ]:
                if _tensor_contains_value(tensor=tensor, value=value):
                    indexes = _tensor_contains_value_at(
                        tensor=tensor, value=value
                    )
                    logger.warning(
                        (
                            "After gathering from all ranks, "
                            "tensor %s contains value %s at indexes %s"
                        ),
                        tensor,
                        value,
                        indexes,
                    )

            # Save results to local tensors
            assignments[i_K] = IGNORE_INDEX
            assignments[i_K][indexes_all] = assignments_all
            indexes[i_K] = IGNORE_INDEX
            indexes[i_K][indexes_all] = indexes_all
            distances[i_K] = float(IGNORE_INDEX)
            distances[i_K][indexes_all] = distances_all
            # For the embeddings, make sure to use j for indexing
            embeddings[i_K][j] = float(IGNORE_INDEX)
            embeddings[i_K][j][indexes_all] = embeddings_all

            j_prev = j
            # next memory bank to use
            j = (j + 1) % len(settings.model.crops_for_assign)

        epoch_comp = epoch + 1

        if (
            # Plot for the first epoch
            epoch_comp == 1
            # Below 100 epochs, plot every 25 epochs,
            or (epoch_comp <= 100 and epoch_comp % 25 == 0)
            # Plot every hundredth epoch
            or epoch_comp % 100 == 0
            # Plot for the last epoch
            or epoch_comp == settings.model.epochs
        ):
            logger.info(
                "Saving clustering data at epoch %s",
                epoch,
            )

            for result, name in [
                (centroids, "centroids.pt"),
                (assignments, "assignments.pt"),
                (distances, "distances.pt"),
                (embeddings, "embeddings.pt"),
                (indexes, "indexes.pt"),
            ]:
                torch.save(
                    result,
                    _create_path(
                        path=settings.dump.tensors,
                        file_name=name,
                        epoch=epoch,
                    ),
                )

            if utils.distributed.is_primary_device():
                # Save which random samples were used as the centroids.
                torch.save(
                    random_idx,
                    _create_path(
                        path=settings.dump.tensors,
                        file_name="centroid-indexes.pt",
                        epoch=epoch,
                    ),
                )
                assignments_cpu = assignments[-1].cpu()
                plotting.embeddings.plot_embeddings_using_tsne(
                    embeddings=embeddings[-1],
                    # Use previous j since this represents which crops
                    # were used for last cluster iteration.
                    crop_index=j_prev,
                    assignments=assignments_cpu,
                    centroids=random_idx,
                    name=f"epoch-{epoch}-embeddings",
                    output_dir=settings.dump.plots,
                )
                plotting.assignments.plot_abundance(
                    assignments=assignments_cpu,
                    name=f"epoch-{epoch}-assignments-abundance",
                    output_dir=settings.dump.plots,
                )
                plotting.assignments.plot_appearance_per_week(
                    assignments=assignments_cpu,
                    name=f"epoch-{epoch}-appearance-per-week",
                    output_dir=settings.dump.plots,
                )
                plotting.autocorrelation.plot_autocorrelation(
                    assignments_cpu,
                    name=f"epoch-{epoch}-autocorrelation",
                    output_dir=settings.dump.plots,
                )
                plotting.autocorrelation.plot_partial_autocorrelation(
                    assignments_cpu,
                    name=f"epoch-{epoch}-partial-autocorrelation",
                    output_dir=settings.dump.plots,
                )
                plotting.transitions.plot_transition_matrix_heatmap(
                    assignments_cpu,
                    name=f"epoch-{epoch}-transition-heatmap",
                    output_dir=settings.dump.plots,
                )
                plotting.transitions.plot_transition_matrix_clustermap(
                    assignments_cpu,
                    name=f"epoch-{epoch}-transition-clustermap",
                    output_dir=settings.dump.plots,
                )

        if utils.distributed.is_primary_device():
            n_unassigned_samples = _calculate_number_of_unassigned_samples(
                assignments[-1],
            )
            percent_unassigned_samples = (
                n_unassigned_samples / assignments.shape[-1] * 100
            )
            logger.warning(
                "Number of unassigned samples: %s (%s%%), assignments.shape=%s",
                n_unassigned_samples,
                round(percent_unassigned_samples, 2),
                assignments.shape[-1],
            )
            if settings.enable_tracking:
                mantik.call_mlflow_method(
                    mlflow.log_metric,
                    "unassigned_samples",
                    n_unassigned_samples,
                    step=epoch,
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
    unassigned_sample_indexes = (assignments == IGNORE_INDEX).nonzero(
        as_tuple=True
    )[0]
    logger.debug("Unassigned indexes: %s", unassigned_sample_indexes)
    n_unassigned_samples = len(unassigned_sample_indexes)
    return n_unassigned_samples


def _tensor_contains_value(tensor: torch.Tensor, value: int | float) -> bool:
    indexes = _tensor_contains_value_at(tensor=tensor, value=value)
    [size] = indexes.size()
    return size != 0


def _tensor_contains_value_at(
    tensor: torch.Tensor, value: int | float
) -> torch.Tensor:
    return (tensor == value).nonzero(as_tuple=True)[0].int()
