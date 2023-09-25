# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import datetime
import logging
import os
import time

import pandas as pd


class LogFormatter:
    def __init__(self, args):
        self.start_time = time.time()
        self.rank = args.global_rank
        self.local_rank = args.local_rank

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "{} - {} - {} - RANK {} (local {})".format(
            record.levelname,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            datetime.timedelta(seconds=elapsed_seconds),
            self.rank,
            self.local_rank,
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return f"{prefix} - {message}" if message else ""


def create_logger(filepath, args):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter(args)

    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a+")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class Stats:
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path
        self.columns = columns

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_csv(self.path, index_col=0)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_csv(self.path)
