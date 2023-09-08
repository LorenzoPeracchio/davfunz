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
you can use the ``relAI.get_random_ingredients()`` function:

.. autofunction:: relAI.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`relAI.get_random_ingredients`
will raise an exception.

.. autoexception:: relAI.InvalidKindError

Build an Autoencoder
----------------

To build an Autoencoder
you can use the ``relAI.create_autoencoder()`` function:



For example:

>>> import relAI
>>> relAI.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

