# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os

import pandas as pd


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
