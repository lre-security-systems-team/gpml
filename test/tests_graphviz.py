# contributors: Pierre Parrend
# project started on 11/10/2022
import sys
import unittest
import pandas as pd

sys.path.append('.')
from gpml.visualisation.graphviz import *
from data.bot_iot.bot_iot_ddos_normal import *
from data.graphviz_config_bot_iot import *


class GraphVizTestCase(unittest.TestCase):

    def setUp(self):
        self.true_value = True
        self.df = pd.read_csv('data/bot_iot/ddos-normal-filter.csv')
        self.graph_type = graph_type
        self.label = label
        self.saddr = saddr
        self.daddr = daddr
        self.sport = sport
        self.dport = dport

    def testMockup(self):
        print_graph(self.df, self.graph_type, self.label, self.saddr, self.daddr, self.sport, self.dport)
        self.assertTrue(self.true_value)
