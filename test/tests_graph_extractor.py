# contributors: Julien MICHEL
# project started on 11/10/2022
import sys
import unittest

sys.path.append('.')
import gpml.data_preparation.graph_extractor as cg


class TestCreateGraph(unittest.TestCase):  # create csv with corresponding function and check csv

    def setUp(self):
        self.filename = 'data/ugr16/ugr_sample_100k.csv'
        self.samplefile = 'data/ugr16/ugr_sample_1k.csv'

    def test_graph_extractor(self):  # test 2 create graph of the same data and check they are the same

        MG1 = cg.create_static_graph_from_csv(self.filename, edge_source=['Source IP'],
                                              edge_dest=['Destination IP'], label='Label', sampling=0.01)
        MG2 = cg.create_static_graph_from_csv(self.samplefile, edge_source=['Source IP'],
                                              edge_dest=['Destination IP'], label='Label')
        self.assertEqual(MG1.number_of_edges(), MG2.number_of_edges())


if __name__ == '__main__':
    unittest.main()
