import functools
import itertools
import logging
import os

import a6

logger = logging.getLogger(__name__)

a6.utils.log_to_stdout()

# Should not need to be changed.
os.environ[
    a6.parallel.slurm.SINGULARITY_IMAGE_ENV_VAR
] = "/p/scratch/deepacf/emmerich1/jupyter-a6/jupyter-kernel.sif"
data_path = (
    "/p/scratch/deepacf/maelstrom/maelstrom_data/4cast-application6"
    "/ml/temperature_level_128_daily_averages_2017_2020.nc"
)

# Set desired queue.
queue = "batch"
project = "deepacf"

# Set desired number of cores.
cores = 16

# Set list of desired nodes to run on.
nodes = []

client = a6.parallel.slurm.JuwelsClient(
    queue=queue,
    project=project,
    cores=cores,
    memory=64,
    processes=4,
    walltime="02:00:00",
    extra_slurm_options=[f"--nodelist {','.join(nodes)}"] if nodes else None,
)

variance_ratio = [None]
n_clusters = [29]
use_varimax = [True]

method = functools.partial(
    a6.benchmark.wrap_benchmark_method_with_logging(a6.pca_and_kmeans),
    data_path,
)
arguments = itertools.product(variance_ratio, n_clusters, use_varimax)

logger.info(
    "Queueing job in %s with account %s requesting %s cores",
    queue,
    project,
    cores,
)

with client.scale(workers=1):
    results = client.execute(method, arguments)

logger.info("Result: %s", results)
