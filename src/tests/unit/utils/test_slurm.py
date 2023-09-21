import pytest

import a6.testing as testing
import a6.utils.slurm as slurm


def test_get_node_id():
    with testing.env.env_vars_set({"SLURM_NODEID": "2"}):
        result = slurm.get_node_id(1)

    assert result == 2


def test_get_node_id_unset():
    result = slurm.get_node_id(1)

    assert result == 1


@pytest.mark.parametrize(
    ("n_nodes", "visible_devices", "expected"),
    [
        (1, None, 1),
        (1, "", 1),
        (1, "1", 1),
        (1, "1,2", 2),
        (2, "1,2", 4),
    ],
)
def test_get_world_size(n_nodes, visible_devices, expected):
    with testing.env.env_vars_set(
        {"SLURM_NNODES": str(n_nodes), "CUDA_VISIBLE_DEVICES": visible_devices}
    ):
        result = slurm.get_world_size()

    assert result == expected
