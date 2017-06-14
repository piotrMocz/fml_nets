"""
The Client module, that primarily takes care of
ingesting the input data and then training the model (MLP),
sending the trained model to the server and, once it gets
the global model back in the response, swaps the current model
for the new one.

From the technical standpoint, the training, sending and
swapping is scheduled as an asynchronous Celery task so
it does not iterrupt the flow of the application.

We use the Celery result backend with active querying
for successful computation on every `predict_next`
request to ensure that we are never doing a busy wait
or blocking the app.

Unlike the server, which is a standalone app, this one is
a library that you import and use as a monitoring tool:
```
    from fml_nets.client import Client

    analytics_client = Client(...)

    def handle_request(...):
        analytics_client.record_request()
        # ....
```
"""
from typing import List

import requests
from datetime import datetime
from keras.models import load_model, Sequential

from collectors import Collector
from learners import MLPLearner
import tasks as tasks

import pandas as pd
import numpy as np

BASE_URL = 'localhost'


class Client:
    """A class encapsulating the client's logic.

    Rest of the heavy lifting is done by the `train_and_send_task`
    """
    def __init__(
            self, name: str, initial_data: pd.DataFrame,
            sampling_rate: str='S', buffer_size: int=2000,
    ):
        self.name = name
        self.url = BASE_URL
        self.collector = Collector(
            data=initial_data,
            sampling_rate=sampling_rate,
            buffer_size=buffer_size,
        )
        self.learner = MLPLearner(initial_data)
        self.pending_learner_result = None

    def record_request(self):
        """As the example problem we are trying to solve with
        this system is time series prediction, this is the method
        used for constructing the time series one-by-one.

        It will produce a single entry in the time series (and
        when you issue more per `sampling_rate`, you will get
        higher bars on the histogram."""
        self.collector.record_request()
        # when we are about to overflow our buffer, schedule learning:
        if self.collector.current_size >= (self.collector.buffer_size - 1):
            self.train()

    def train(self):
        """Schedule the training of the model.

        This method does not block, it collects the data
        and let's the Celery task to the computations."""
        data = self.collector.get_data()
        if isinstance(data, pd.Series):
            data = pd.DataFrame({'hits': data.values}, index=data.index)

        # schedule async execution of the training:
        self.pending_learner_result = tasks.train_and_send_task.delay(
            self.name, self.url, self.learner, data,
        )

    def predict_next(self) -> np.float:
        """Get the next element in the time series.

        Can be used to produce longer sequences when
        used repeatedly on its own results."""
        if (self.pending_learner_result and
                self.pending_learner_result.successful()):
            self.learner = self.pending_learner_result.get()
            self.pending_learner_result = None

        # TODO: this prediction is for the look-back methods, change:
        seq = self.collector.buffer.data['hits'].values[
            -self.learner.look_back:
        ]
        return self.learner.predict_next(seq)
