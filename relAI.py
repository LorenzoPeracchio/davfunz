"""
relAI - Python library for the assessment of the pointwise reliability of ML predictions.
"""

__version__ = "0.1.0"

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn import tree

import plotly.graph_objects as go
import matplotlib.pyplot as plt

class InvalidKindError(Exception):
    """Raised if the kind is invalid."""
    pass


def get_random_ingredients(kind=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise relAI.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]

class CosineActivation(torch.nn.Module):
    """
    A custom activation function that applies the cosine function.

    The CosineActivation class is a PyTorch module that applies the cosine activation function
    to the input tensor.

    """

    def __init__(self):
        """
        Initializes an instance of the CosineActivation class.

        """
        super().__init__()

    def forward(self, x):
        """
        Applies the cosine activation to the input tensor.

        The forward method takes an input tensor and applies the cosine activation function
        element-wise, subtracting the input value from its cosine.

        :param torch.Tensor x: The input tensor.

        :return: The tensor with the cosine activation applied.
        :rtype: torch.Tensor

        """
        return torch.cos(x) - x


class AE(torch.nn.Module):
    """
    Autoencoder model implemented as a PyTorch module.

    The AE class represents an autoencoder model with specified sizes of the layers.
    It consists of an encoder and a decoder, both utilizing the CosineActivation as
    the activation function.

    :param list[int] layer_sizes: A list containing the sizes of the layers of the encoder (decoder built with symmetry).

    :ivar torch.nn.Sequential encoder: The encoder module.
    :ivar torch.nn.Sequential decoder: The decoder module.

    """

    def __init__(self, layer_sizes):
        """
        Initializes an instance of the AE class.

        :param list[int] layer_sizes: A list of integers containing the sizes of the layers.
        """
        super().__init__()
        self.encoder = self.build_encoder(layer_sizes)
        self.decoder = self.build_decoder(layer_sizes)

    def build_encoder(self, layer_sizes):
        """
        Builds the encoder part of an autoencoder model based on the specified layer sizes.

        :param list[int] layer_sizes: A list of integers representing the number of nodes in each layer of the encoder.

        :return: The encoder module of the autoencoder model.
        :rtype: torch.nn.Sequential
        """
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(CosineActivation())
        return torch.nn.Sequential(*encoder_layers)

    def build_decoder(self, layer_sizes):
        """
        Builds the decoder part of an autoencoder model based on the specified layer sizes.

        :param list[int] layer_sizes: A list of integers representing the number of nodes in each layer of the decoder.

        :return: The decoder module of the autoencoder model.
        :rtype: torch.nn.Sequential
        """
        decoder_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            decoder_layers.append(CosineActivation())
        return torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Performs the forward pass of the autoencoder model.

        The forward method takes an input tensor and passes it through the encoder,
        obtaining the encoded representation. The encoded representation is then passed
        through the decoder to reconstruct the original input.

        :param torch.Tensor x: The input tensor.

        :return: The reconstructed tensor.
        :rtype: torch.Tensor
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ReliabilityDetector:
    """
    Reliability Detector for assessing the reliability of data points.

    The ReliabilityDetector class computes the reliability of data points based on
    a specified autoencoder (ae), a proxy model (clf), and an MSE threshold (mse_thresh).

    :param AE ae: The autoencoder model.
    :param proxy_model: The proxy model used for the local fit reliability computation.
    :param float mse_thresh: The MSE threshold used for the density reliability computation.

    :ivar AE ae: The autoencoder model.
    :ivar clf: The proxy model used for the local fit reliability computation.
    :ivar float mse_thresh: The MSE threshold for the density reliability computation.

    """

    def __init__(self, ae, proxy_model, mse_thresh):
        """
        Initializes an instance of the ReliabilityDetector class.

        :param AE ae: The autoencoder model.
        :param proxy_model: The proxy model used for the local fit reliability computation.
        :param float mse_thresh: The MSE threshold used for the density reliability computation.
        """
        self.ae = ae
        self.clf = proxy_model
        self.mse_thresh = mse_thresh

    def compute_density_reliability(self, x):
        """
        Computes the density reliability of a data point.

        The density reliability is determined by computing the mean squared error (MSE)
        between the input data point and its reconstructed representation obtained from
        the autoencoder. If the MSE is less than (or equal to) the specified MSE threshold,
        the data point is considered reliable (returns 1), otherwise unreliable (returns 0).

        :param numpy.ndarray x: The input data point.

        :return: The density reliability value (1 for reliable, 0 for unreliable).
        :rtype: int
        """
        mse = mean_squared_error(x, self.ae((torch.tensor(x)).float()).detach().numpy())
        return 1 if mse <= self.mse_thresh else 0

    def compute_localfit_reliability(self, x):
        """
        Computes the local fit reliability of a data point.

        The local fit reliability is determined by using the proxy model to predict the local fit
        reliability of the input data point. The input data point is reshaped to match the
        expected input format of the proxy model. The predicted reliability value is returned.

        :param numpy.ndarray x: The input data point.

        :return: The local fit reliability class predicted by the proxy model (1 for reliable, 0 for unreliable).
        :rtype: int
        """
        return self.clf.predict(x.reshape(1, -1))[0]

    def compute_total_reliability(self, x):
        """
        Computes the combined reliability of a data point.

        The combined reliability is determined by combining the density reliability and the
        local fit reliability. If both reliabilities are positive (1), the data point is
        considered reliable (returns True), otherwise unreliable (returns False).

        :param numpy.ndarray x: The input data point.

        :return: The combined reliability value (True for reliable, False for unreliable).
        :rtype: bool
        """
        density_rel = self.compute_density_reliability(x)
        localfit_rel = self.compute_localfit_reliability(x)
        return density_rel and localfit_rel


class DensityPrincipleDetector:
    """
    Density Principle Detector for assessing the density reliability of data points.

    The DensityPrincipleDetector class computes the density reliability of data points based on
    a specified autoencoder (autoencoder) and a threshold (threshold).

    :param AE autoencoder: The autoencoder model.
    :param float threshold: The threshold for determining the density reliability.

    :ivar AE ae: The autoencoder model.
    :ivar float thresh: The threshold for determining the density reliability.
    
    """

    def __init__(self, autoencoder, threshold):
        """
        Initializes an instance of the DensityPrincipleDetector class.

        :param AE autoencoder: The autoencoder model.
        :param float threshold: The threshold for determining the density reliability.
        """
        self.ae = autoencoder
        self.thresh = threshold

    def compute_reliability(self, x):
        """
        Computes the density reliability of a data point.

        The density reliability is determined by computing the mean squared error (MSE)
        between the input data point and its reconstructed representation obtained from
        the autoencoder. If the MSE is less than or equal to the specified threshold,
        the data point is considered reliable (returns 1), otherwise unreliable (returns 0).

        :param numpy.ndarray x: The input data point.

        :return: The density reliability value (1 for reliable, 0 for unreliable).
        :rtype: int
        """
        mse = mean_squared_error(x, self.ae((torch.tensor(x)).float()).detach().numpy())
        return 1 if mse <= self.thresh else 0

