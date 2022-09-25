import argparse
import functools
import itertools
import logging
import os
import sys

import a6
import joblib
import numpy as np


logger = logging.getLogger(__name__)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

# prepare the logger
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--profile",
    help="Name of IPython profile to use",
)
args = parser.parse_args()
profile = args.profile
logging.basicConfig(
    filename=os.path.join(FILE_DIR, profile + ".log"),
    filemode="w",
    level=logging.DEBUG,
)
logger.info("number of CPUs found: {0}".format(joblib.cpu_count()))
logger.info("args.profile: {0}".format(profile))

n_workers = int(os.environ.get("N_WORKERS", 1))

data_path = (
    "/home/fabian/Documents/MAELSTROM/data/"
    "pca/temperature_level_128_daily_averages_2020.nc"
)
variance_ratio = np.arange(0.94, 0.96, 0.1)
n_clusters = np.arange(2, 4)
use_varimax = [False]
method = functools.partial(
    a6.pca_and_kmeans,
    data_path,
)
arguments = itertools.product(variance_ratio, n_clusters, use_varimax)
result = a6.parallel.ipyparallel.execute_parallel(
    method=method,
    args=arguments,
    ipython_profile=profile,
    n_workers=n_workers,
    working_directory=FILE_DIR,
)
logger.info("Result: %s", result)
