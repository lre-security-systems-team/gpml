# contributors: Majed JABER
# project started on 11/10/2022

import sys
import unittest

import pandas as pd

sys.path.append('.')
from gpml.data_preparation.time_series_extractor import extract_time_series
from gpml.metrics.spectral_metrics import get_first_value, extract_spectral_metrics


class TestSpectral(unittest.TestCase):
    def setUp(self):
        root = '.'
        self.filename = 'data/bot_iot/UNSW_2018_IoT_Botnet.csv'
        self.df = pd.read_csv(root + '/' + self.filename)

    def test_get_first_value(self):
        series = self.df['pkts']
        first_value = get_first_value(series)
        assert first_value == series.iloc[0]

    def test_spectral_metrics_extractor(self):
        features_list = ['stime', 'datetime', 'saddr', 'daddr', 'sport', 'dport', 'pkts', 'bytes', 'rate', 'attack',
                         'category', 'subcategory', 'weight', 'dur', 'mean', 'sum', 'min', 'max', 'spkts', 'dpkts',
                         'srate',
                         'drate']
        sortby_list = ['stime']
        groupby_list = ['stime', 'datetime', 'saddr', 'daddr']
        aggregation_dict = {
            'pkts': 'sum',
            'bytes': 'sum',
            'attack': 'first',
            'category': 'first',
            'subcategory': 'first',
            'rate': 'mean',
            'dur': 'mean',
            'mean': 'mean',
            'sum': 'mean',
            'min': 'mean',
            'max': 'mean',
            'spkts': 'mean',
            'srate': 'mean',
            'drate': 'mean',
            'weight': 'sum'
        }

        result = extract_time_series(self.df, 'stime', 's', features_list, sortby_list, groupby_list,
                                           aggregation_dict)

        stime = 'stime'
        saddr = 'saddr'
        daddr = 'daddr'
        pkts = 'pkts'
        bytes = 'bytes'
        rate = 'rate'
        lbl_category = 'attack'

        result = extract_spectral_metrics(result, stime, saddr, daddr, pkts, bytes, rate, lbl_category)
        print('res:', result)
        assert not result.empty
        assert 'ts1_m1_pkts' in result.columns
        assert 'ts2_m1_bytes' in result.columns
        assert 'attack' in result.columns
