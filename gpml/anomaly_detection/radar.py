"""
This module provides an implementation of RADAR.

Li, J., Dani, H., Hu, X., & Liu, H. (2017, August).
Radar: Residual analysis for anomaly detection in attributed networks.
In IJCAI (Vol. 17, pp. 2152-2158).

It includes a class to process NetworkX graphs and numpy arrays for attributes
to retrieve anomalies.

Class:
    RADAR - RADAR Algorithm class

Public Methods:
    fit - feed data to the RADAR object and compute anomalies.
    get_score - returns the anomalies

Contributors:
    Lyes BOURENNANI

Project started on:
    11/10/2022
"""

import tensorflow as tf
import networkx as nx
import numpy as np
from numpy.linalg import inv, norm
from gpml.anomaly_detection.detector import Detector

class RADAR(Detector):
    """
    RADAR Class.

    Can be instantiated with the studied graph and attributed matrix.
    It directly stores the result of the fit() method.
    Anomaly scores are retrieved using get_score() method.
    """

    def fit(self, alpha: float, beta: float, gamma: float, epochs: int = 100, verbose: bool=False):
        """
        Compute the anomaly scores with the given algorithm parameter.

        It uses the Adam optimizer instead of the paper optimization algorithm.

        Returns
        -------
        output void

        Parameters
        ----------
        :param alpha: Parameter alpha of the algorithm
        :param beta: Parameter beta of the algorithm
        :param gamma: Parameter gamma of the algorithm
        :param epochs: Number of optimization epoch (Defaults to 100)
        :param verbose: Boolean for verbose (Defaults to False)
        """
        if self.graph is None or len(self.graph.nodes()) <= 0:
            self.score = None
            return

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.004)

        def loss_function():
            return tf.norm(X - (X @ tf.transpose(W) @ X) - R, ord='fro', axis=(0,1)) \
                + alpha * self.l21_norm(W) \
                + beta * self.l21_norm(R) \
                + gamma * tf.linalg.trace(tf.transpose(R) @ L @ R)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss = loss_function()
                gradients = tape.gradient(loss, [W, R])
                optimizer.apply_gradients(zip(gradients, [R]))
            return loss

        A = tf.Variable(nx.adjacency_matrix(self.graph).todense(), dtype=tf.float32)
        X = tf.Variable(np.array(self.attributes), dtype=tf.float32)
        n = A.shape[0]

        L = nx.laplacian_matrix(self.graph).todense()

        W = tf.Variable(np.eye(n), dtype=tf.float32)
        R = tf.Variable(inv((beta + 1) * np.eye(n) + gamma * L) @ X, dtype=tf.float32)

        for epoch in range(epochs):
            loss = train_step()

            if verbose:
                print(f'Epoch {epoch + 1}: objective equation value =  {loss}')

        score = norm(R, 2, axis=1)
        self.score = {node: score[idx] for idx, node in enumerate(self.graph.nodes())}
