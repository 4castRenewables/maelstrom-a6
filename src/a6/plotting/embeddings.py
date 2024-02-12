import logging
import pathlib
import time
from collections.abc import Iterator

import matplotlib.pyplot as plt
import torch

import a6.plotting._colors as _colors

logger = logging.getLogger(__name__)


def plot_embeddings_using_tsne(
    embeddings: torch.Tensor,
    crop_index: int,
    assignments: torch.Tensor,
    centroids: torch.Tensor,
    name: str = "embeddings",
    output_dir: pathlib.Path | None = None,
) -> None:
    """Plot the embeddings of DCv2 using t-SNE.

    Args:
        embeddings (torch.Tensor, shape(n_crops, n_samples, n_embedding_dims)):
            Embeddings as produced by the ResNet.
        crop_index (int): Crop index.
        assignments (torch.Tensor, shape(n_samples)):
            Assignments by DCv2 for each sample.

            Used for coloring each sample in the plot.
        centroids (torch.Tensor): The indexes of the centroids.
        name (str): Name of the figure
        output_dir (pathlib.Path): Path where to save the figure.

    """
    import openTSNE

    start = time.time()
    logger.info("Creating plot for embeddings")

    _, ax = plt.subplots()

    ax.set_title(f"Embeddings for crops {crop_index}")

    (x, y), (x_centroids, y_centroids) = _fit_tsne(
        embeddings=embeddings[crop_index], centroids=centroids
    )
    colors = _colors.create_colors_for_assigments(assignments)

    ax.scatter(x, y, c=colors, s=1)
    ax.scatter(x_centroids, y_centroids, c="red", s=20, marker="x")

    if output_dir is not None:
        plt.savefig(output_dir / f"{name}-crops-{crop_index}.pdf")

    logger.info("Finished embeddings plot in %s seconds", time.time() - start)


def _fit_tsne(
    embeddings: torch.Tensor, centroids: torch.Tensor
) -> Iterator[tuple[tuple[float, float], tuple[float, float]]]:
    start = time.time()
    logger.info("Fitting t-SNE")

    result = openTSNE.TSNE().fit(embeddings.cpu())

    logger.info("Finished fitting t-SNE in %s seconds", time.time() - start)

    logger.info(
        "embeddings: %s centroids: %s result: %s",
        embeddings.size(),
        centroids,
        result.shape,
    )

    return zip(*result), zip(*result[centroids])
