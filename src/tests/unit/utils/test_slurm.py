import a6.testing as testing
import a6.utils.slurm as slurm


def test_get_node_id():
    with testing.env.env_vars_set({"SLURM_NODEID": "2"}):
        result = slurm.get_node_id()

    assert result == "2"
