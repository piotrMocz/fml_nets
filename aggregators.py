from typing import List

import numpy as np
import pandas as pd


class Aggregator:
    def __init__(self, sample_rate: str='S'):
        self.sample_rate = sample_rate

    def aggregate(self, batches: List[pd.Series]) -> pd.Series:
        """Combines arbitrarily many time series into one,
        summing up the hits"""
        mini = np.min([b.index.min() for b in batches])
        maxi = np.max([b.index.max() for b in batches])

        # create a date range covering all the series
        date_rng = pd.date_range(start=mini, end=maxi, freq=self.sample_rate)
        hits = np.zeros(len(date_rng))
        ts = pd.Series(data=hits, index=date_rng)

        for b in batches:
            ts = ts.combine(b, func=np.add, fill_value=0)

        return ts
