# contributors: Majed JABER
# project started on 11/10/2022

import sys
import unittest

import networkx as nx
import pandas as pd

sys.path.append('.')
from gpml.metrics.spectral_metrics import extract_spectrum


class TestSpectral(unittest.TestCase):
    def setUp(self):
        root = '.'
        self.filename = 'data/bot_iot/UNSW_2018_IoT_Botnet.csv'
        self.df = pd.read_csv(root + '/' + self.filename)

    def test_extract_spectrum(self):
        tw = self.df.head(100000)
        g, l, ev = extract_spectrum(tw, 'saddr', 'daddr', 'pkts')
        assert isinstance(g, nx.Graph)
        assert l.shape[0] == l.shape[1] == g.number_of_nodes()
        assert len(ev) == g.number_of_nodes()
