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

