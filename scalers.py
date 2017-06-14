import numpy as np


class Scaler(object):
    def __init__(self, predicted_col=None):
        self.predicted_col = predicted_col

    def fit_transform(self, batch):
        pass


class NaiveScaler(Scaler):
    def __init__(self, predicted_col):
        super(NaiveScaler, self).__init__(predicted_col)

    def fit_transform(self, batch):
        batch = batch.dropna()
        # TODO: handle this SettingWithCopyWarning thing
        # scaling is easy, as we know the minimums and maximums:
        batch['minute'] /= 60.0
        batch['hour'] /= 24.0
        batch['day'] /= 30.0
        batch['weekday'] /= 7.0
        batch['month'] /= 12.0
        batch['year'] /= 2010.0
        batch[self.predicted_col] = np.log1p(batch[self.predicted_col])
        return batch


class SKOSScaler(Scaler):
    def __init__(self, predicted_col='hits'):
        super(SKOSScaler, self).__init__(predicted_col)
        self.predict_max = None

    def fit_transform(self, batch):
        batch['hour'] /= 24
        batch['day'] /= 31.0
        batch['weekday'] /= 7.0
        # batch['month'] = batch['month'] / 12.0
        # batch['year'] = batch['year'] / 2017.0
        self.predict_max = batch[self.predicted_col].max()
        batch[self.predicted_col] = batch[self.predicted_col] / batch[self.predicted_col].max()

        return batch

    def inverse_transform(self, batch):
        return batch * self.predict_max
