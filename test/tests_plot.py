# contributors: Julien MICHEL
# project started on 11/10/2022

import sys

sys.path.append('.')
import unittest
import networkx as nx
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import scipy
import matplotlib.cm as cm
import time
from statistics import mean
import timeit
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gpml.visualisation.plot as gplt
from xgboost import XGBClassifier


class Test_plot(unittest.TestCase):

    def setUp(self):
        root = '.'
        self.filename = 'data/ugr16/ugr_sample_100k.csv'
        self.df = pd.read_csv(root + '/' + self.filename)

    def test_plot_importance(self):
        df_original_data = self.df.copy()
        binary_label = 'Label'
        attack_cat_label = 'category'
        attack_label = 'category'
        df_original_data['category'] = df_original_data[binary_label].copy()

        df_original_data[binary_label].replace(["background", "blacklist"], 0, inplace=True)
        df_original_data[binary_label].replace(["nerisbotnet", "dos", "scan44", "scan11", "anomaly-spam"], 1,
                                               inplace=True)
        attacks = df_original_data[binary_label].unique()
        categories = df_original_data[attack_cat_label].unique()
        subcategories = df_original_data[attack_label].unique()

        # Create a mapping dictionary
        map_dict = {
            'background': 0,
            'blacklist': 0,
            'anomaly-spam': 1,
            'dos': 2,
            'scan11': 3,
            'scan44': 4,
            'nerisbotnet': 5
        }
        # Replace the values in the 'subcategory' column using the mapping dictionary
        targets_labels = df_original_data[attack_label]
        df_for_learning = df_original_data.copy()
        df_for_learning[attack_label] = df_for_learning[attack_label].replace(map_dict)

        is_attack = df_for_learning[binary_label]
        category_label = df_for_learning[attack_cat_label]
        targets_num = df_for_learning[attack_label]

        df_for_learning = df_for_learning.drop(labels=[binary_label, attack_cat_label, attack_label], axis=1)

        df_numerical_data = df_for_learning.select_dtypes(np.number)
        df_categorical_data = df_for_learning.select_dtypes(exclude=np.number)

        clf = XGBClassifier(tree_method="hist", device="cuda")
        clf = clf.fit(df_numerical_data, targets_num)

        gplt.plot_high_gain_features(clf, 'test', df_numerical_data, attack_label, targets_labels)
        self.assertTrue(True)
        # gplt.plot_importance(clf, 'test')
        self.assertTrue(True)
