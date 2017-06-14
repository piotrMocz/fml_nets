import pandas as pd


class SKOSWrapper(object):
    def __init__(self, path: str='data/skos/', file: str='skos_agg.csv',
                 nrows: int=2000, sample_rate: str='H'):
        df_skos = pd.read_csv(
            path+file, header=None, index_col=0, names=['time', 'hits'],
            converters={
                'time': lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
            }, sep=',', nrows=nrows)
        self.agg_df = df_skos.resample(sample_rate).sum()
        self.agg_df.fillna(0, inplace=True)

    def extend_df(self):
        self.agg_df['hour'] = self.agg_df.index.map(lambda t: t.hour)
        self.agg_df['day'] = self.agg_df.index.map(lambda t: t.day)
        self.agg_df['weekday'] = self.agg_df.index.map(lambda t: t.weekday)
        # self.agg_df['month'] = self.agg_df.index.map(lambda t: t.month)
        # self.agg_df['year'] = self.agg_df.index.map(lambda t: t.year)

    def scale(self):
        self.agg_df['hour'] = self.agg_df['hour']/23.0
        self.agg_df['day'] = self.agg_df['day'] / 31
        self.agg_df['weekday'] = self.agg_df['weekday'] / 7.0
        self.max_hits = self.agg_df['hits'].max()
        self.agg_df['hits'] = self.agg_df['hits'] / self.max_hits

    def inv_scale(self, data):
        return data*self.max_hits

    def get_df(self) -> pd.DataFrame:
        return self.agg_df

    def get_processed_df(self, scale=True) -> pd.DataFrame:
        """Perform all the necessary processing and return the aggregated data
        ready for processing"""
        self.extend_df()
        if scale:
            self.scale()
        return self.get_df()
