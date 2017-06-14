from datetime import datetime

import numpy as np
import pandas as pd

from utils import Queue


class Collector:
    """Runs with a web application and can be notified of the incoming requests.

    To record the request, use the `record_request` method.
    In order to make the collection work as part of the predictor
    system, call `register_collector` method on the main `Runner` object.
    """
    id = 0

    def __init__(self, data: np.array=None, sampling_rate: str='S',
                 buffer_size: int=2000):
        self.id = Collector.id
        Collector.id += 1
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.buffer = Queue[np.float](data=data, size_limit=buffer_size)
        self.batch = None

    def aggregate_batch(self) -> pd.Series:
        """Returns a time series with the hits per
        given (by sampling_rate) interval"""
        buff = self.buffer.data
        hits = np.ones(len(buff))
        ts = pd.Series(data=hits, index=buff)
        ts = ts.resample(self.sampling_rate).sum().fillna(0)
        self.batch = ts
        return ts

    def get_data(self) -> pd.Series:
        """Gets the aggregated data and cleans the buffer"""
        self.aggregate_batch()
        self.buffer.clean()

        return self.batch

    def record_request(self):
        """Record a single http request"""
        self.buffer.push(datetime.now())

    @property
    def current_size(self):
        return self.buffer.size
