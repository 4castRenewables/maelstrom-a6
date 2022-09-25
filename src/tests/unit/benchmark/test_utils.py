import typing as t
import unittest.mock as mock

import a6.benchmark as bench
from dask.delayed import Delayed

# Cannot be imported otherwise,
# because dask mixes up function delayed and submodule delayed"""


@mock.patch("logging.getLogger")
def test_wrap_benchmark_method_with_loggin(mock_get_logger):
    wrapped = bench.utils.wrap_benchmark_method_with_logging(lambda x: x)
    assert wrapped("A") == "A"
    mock_get_logger.assert_called()


def test_make_method_lazy():
    lazy_print = bench.utils.make_method_lazy(print)
    result = lazy_print("A")
    assert isinstance(lazy_print, t.Callable)
    assert isinstance(result, Delayed)
