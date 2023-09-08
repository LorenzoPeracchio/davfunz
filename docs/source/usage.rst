Usage
=====

.. _installation:

Installation
------------

To use relAI, first install it using pip:

.. code-block:: console

   (.venv) $ pip install relAI

relAI
-----

.. autofunction:: relAI.compute_dataset_avg_mse

.. autofunction:: relAI.compute_dataset_reliability

.. autofunction:: relAI.create_autoencoder

.. autofunction:: relAI.create_and_train_autoencoder

.. autofunction:: relAI.create_reliability_detector

.. autofunction:: relAI.density_predictor

.. autofunction:: relAI.generate_synthetic_points

.. autofunction:: relAI.mse_threshold_barplot

.. autofunction:: relAI.mse_threshold_plot

.. autofunction:: relAI.perc_mse_threshold

.. autofunction:: relAI.train_autoencoder


For example:

>>> import relAI
>>> layer_sizes = [1, 2, 3]
>>> relAI.create_autoencoder(layer_sizes)
['shells', 'gorgonzola', 'parsley']

