import math
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from scalers import NaiveScaler, Scaler


NPArrayPair = Tuple[np.ndarray, np.ndarray]
ScalerType = Union[Scaler, TransformerMixin]


class OnlineLearner:
    """Base class for all online learners."""
    def __init__(self, data: Optional[pd.DataFrame],
                 predicted_col: str='hits',
                 scaler: Optional[ScalerType]=None):
        """Initialize with base data.

        :param data: pandas DataFrame with data (both X and y)
        :param predicted_col: the name of the column to predict
        :param scaler: a `Scaler` instance that will be used to scale features
        """
        self.model = None
        self.predicted_col = predicted_col

        # this is a trick: we want to maintain the type hierarchy,
        # but if it's a TimeWindowLearner, we would like not to call
        # stuff below
        if data is None:
            return

        if scaler is None:
            self.scaler = NaiveScaler(predicted_col)
        else:
            self.scaler = scaler
            self.scaler.predicted_col = predicted_col

        self.input_dim = len(data.columns) - 1
        self.X, self.y = self.prepare_batch(data)

    def prepare_batch(self, batch: pd.DataFrame) -> NPArrayPair:
        """Do basic preprocessing of the batch and split it (X and y)."""
        batch = self.scaler.fit_transform(batch)
        y = batch[self.predicted_col].values
        X = batch.drop(self.predicted_col, axis=1).values
        return X, y

    def create_model(self):
        """Create and initialize the model."""
        pass

    def train_on_batch(self, batch: pd.DataFrame):
        """Train the model on one batch of data."""
        pass

    def predict_from_batch(self, batch: pd.DataFrame):
        """Predict the results from a batch of data.

        For now the assumption is that the batch contains
        the predicted column (with valid data or not) and
        just ignores it. But it is necessary to supply it,
        so that the dimensions agree."""
        pass

    def predict_next(self, points: np.array) -> np.array:
        pass

    def test_on_batch(self, batch: pd.DataFrame):
        """Test predict on the batch and compare
        with actual data.

        Draws a comparative plot."""
        pass


class MLPLearner(OnlineLearner):
    """A simple learner, using an MLP to predict the time series."""
    def __init__(self, data: pd.DataFrame, predicted_col: str='hits',
                 scaler: Optional[ScalerType]=None):
        super().__init__(data, predicted_col, scaler)

    def create_model(self):
        self.model = Sequential()
        # adjust the number of layers and neurons
        self.model.add(Dense(6, input_dim=self.input_dim, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        # adjust the optimizer, regularizations, etc.
        self.model.compile(optimizer='sgd', loss='mse')
        self.model.fit(self.X, self.y, batch_size=50)
        return self

    def train_on_batch(self, batch: pd.DataFrame):
        X, y = self.prepare_batch(batch)
        self.model.train_on_batch(X, y)

    def predict_from_batch(self, batch: pd.DataFrame) -> np.ndarray:
        X, y = self.prepare_batch(batch)
        predictions = self.model.predict_on_batch(X)
        return predictions

    def serialize_model(self, name: str):
        self.model.save(name)
