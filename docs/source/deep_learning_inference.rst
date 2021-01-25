Deep Learning - Inference
=========================

Loading model
-------------
To load a pretrained network one should use :code:`atlalign.ml_utils.load_model`.
Note that it loads all possible custom layers in the background so that the user does not have
to worry about it.

.. code-block:: python

    from atlalign.ml_utils import load_model

    path_1 = 'path/to/a/folder'  # inside of this folder .json (architecture) and .h5 (weights)
    path_2 = 'path/to/a/file.h5'  # architecture and weights not separated + additional info (loss and optimizer)

    model_path_1 = load_model(path_1)
    model_path_2 = load_model(path_2, compile=True)

Merging
-------
To merge a global and a local alignment network one can perform the composition via the :code:`__call__` method
of :code:`atlalign.base.DisplacementField` on a per sample basis. A better approach is to use a custom keras layer
implementing composition. For the latter option we provide a utility function :code:`atlalign.ml_utils.merge_global_local`.

.. code-block:: python

    from atlalign.ml_utils import load_model, merge_global_local

    path_global = 'global_model.h5'
    path_local = 'local_model.h5'

    model_global = load_model(path_global)
    model_local = load_model(path_local)

    model_merged = merge_global_local(model_global, model_local)

Forward pass
------------
Performing the actual inference is extremely simple. Please review the :ref:`dl_training.supervised_generator` to
understand the shape of the expected input. To quickly summarize (for non-inverse models) the user only needs to create
a 4D array of the following shape

.. code-block:: python

    (batch_size, height=320, width=456, depth=2)

The last dimension is simply a stack of the :code:`img_ref` and :code:`img_mov`.

.. code-block:: python

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # to enable GPU

    import numpy as np

    from atlalign.base import DisplacementField
    from atlalign.ml_utils import load_model

    batch_size = 32  # how many samples are grouped together at inference time

    model = load_model('path/to/model.h5')
    X = np.random.random((200, 320, 456, 2))

    [reg_images, deltas_xy] = model.predict(X, batch_size=batch_size)

    # one can also create instances of DisplacementFields to perform many other tasks
    dfs = [DisplacementField(deltas_xy[i, ..., 0], deltas_xy[i, ..., 1]) for i in range(len(X))]
