Usage
=====

.. _installation:

Installation
------------

To use relAI, first install it using pip:

.. code-block:: console

   (.venv) $ pip install relAI

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``create_autoencoder()`` function:

.. autofunction:: relAI.create_autoencoder

For example:

>>> import relAI
>>> layer_sizes = [1, 2, 3]
>>> relAI.create_autoencoder(layer_sizes)
['shells', 'gorgonzola', 'parsley']

