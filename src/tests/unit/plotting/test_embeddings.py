import torch

import a6.plotting.embeddings as _embeddings


def test_plot_embedding(tmp_path):
    # Embeddings for 2 crops, 2 samples, and embedding dims of size 4
    embeddings = torch.Tensor(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
        ]
    )

    # Assignments for 2 samples
    assignments = torch.Tensor(
        [1, 2],
    )

    centroids = torch.Tensor(
        [0, 1],
    ).int()

    expected = tmp_path / "test-crops-0.pdf"

    _embeddings.plot_embeddings_using_tsne(
        embeddings=embeddings,
        crop_index=0,
        assignments=assignments,
        centroids=centroids,
        name="test",
        output_dir=tmp_path,
    )

    assert expected.exists()
