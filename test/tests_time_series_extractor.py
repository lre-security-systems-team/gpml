# contributors: Majed JABER
# project started on 11/10/2022

import sys
import unittest
import pandas as pd

sys.path.append('.')
from gpml.data_preparation.time_series_extractor import extract_time_series


class TestSpectral(unittest.TestCase):
    def setUp(self):
        root = '.'
        self.filename = 'data/bot_iot/UNSW_2018_IoT_Botnet.csv'
        self.df = pd.read_csv(root + '/' + self.filename)

    def test_time_series_extractor(self):
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

        result = extract_time_series(self.df, 'stime', 's', features_list, sortby_list, groupby_list
                                       , aggregation_dict)
        print(result)
        assert not result.empty
        assert 'datetime' in result.columns
        assert result['pkts'].sum() == self.df['pkts'].sum()
