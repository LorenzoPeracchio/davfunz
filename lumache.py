"""
Lumache - Python library for cooks and food lovers.
"""

__version__ = "0.1.0"


class InvalidKindError(Exception):
    """Raised if the kind is invalid."""
    pass


def get_random_ingredients(kind=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]

def create_autoencoder(layer_sizes):
    """
    Gets an autoencoder model with the specified sizes of the layers.

    :param layer_sizes: A list containing the number of nodes of each layer of the encoder (decoder built with symmetry).
    :type layer_sizes: list[int]

    :return: An instance of the autoencoder model.
    :rtype: list[int]
    """
    ae = AE(layer_sizes)
    return ae
