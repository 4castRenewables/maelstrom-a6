import functools
import itertools
import logging
import os

import lifetimes

logger = logging.getLogger(__name__)

lifetimes.utils.log_to_stdout()

# Change to correct image path.
os.environ[lifetimes.parallel.slurm.SINGULARITY_IMAGE_ENV_VAR] = os.environ.get(
    "SINGULARITY_IMAGE"
)

# Should not need to be changed.
data_path = (
    "/data/maelstrom/a6/temperature_level_128_daily_averages_2017_2020.nc"
)

# Set desired queue.
queue = "casc-lw"
project = "maelstrom"

# Set desired number of cores.
cores = 16

# Set list of desired nodes to run on.
nodes = []

client = lifetimes.parallel.slurm.E4Client(
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
    lifetimes.benchmark.wrap_benchmark_method_with_logging(
        lifetimes.pca_and_kmeans
    ),
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
