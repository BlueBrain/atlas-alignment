.. _datasets:

Datasets
========

We include multiple dataset loading utilities. Note that for some functions 
implemented in the :code:`atlalign.data` module we assume that the user has 
the underlying raw data locally.

Nissl volume
------------
Contains the 25 microns reference volume used within the entire project. The corresponding function
is :code:`nissl_volume` that returns a (528, 320, 456, 1) numpy array.


Annotation volume
-----------------
Contains per pixel segmentation/annotation of the entire reference atlas. The corresponding
function is :code:`annotation_volume` that returns a (528, 320, 456) numpy array. Note
that one can use the corresponding :code:`segmentation_collapsing_labels` function
to get the tree-like hierarchy of the classes.


Manual registration
-------------------
This dataset comes from the labeling tool that extracts displacement fields. The corresponding function is
:code:`manual_registration`.
Available genes:

- **Calb1**
- **Calb2**
- **Cck**
- **Npy**
- **Pvalb**
- **Sst**
- **Vip**

.. code-block:: python

    from atlalign.data import manual_registration

    res = manual_registration()

    assert set(res.keys()) == {'dataset_id', 'deltas_xy', 'image_id', 'img', 'inv_deltas_xy', 'p'}
    assert len(res['image_id']) == 278

The returned dictionary contains the following keys:

 - :code:`dataset_id` - unique id of the section dataset
 - :code:`deltas_xy` - array of shape (320, 456, 2) where the last dimension represents the x (resp y) deltas of the transformation
 - :code:`image_id` - unique id of the section image
 - :code:`img` - moving image of shape (320, 456) that was preregistered with the Allen API
 - :code:`inv_deltas_xy` - same as :code:`deltas_xy` but represents the inverse transformation
 - :code:`p` - coronal coordinate in microns [0, 13200]

To perform the registration instantiate :code:`atlalign.base.DisplacementField` using the :code:`deltas_xy` and warp the
:code:`img` with it.

.. code-block:: python

    from atlalign.base import DisplacementField
    from atlalign.data import manual_registration

    import numpy as np

    res = manual_registration()
    i = 10
    delta_x = res['deltas_xy'][i, ..., 0]
    delta_y = res['deltas_xy'][i, ..., 1]
    img_mov = res['img'][i]

    df = DisplacementField(delta_x, delta_y)
    img_reg = df.warp(img_mov)

For more details on :code:`atlalign.base.DisplacementField` see :ref:`building_blocks`.

Dummy
-----
Artificially generated datasets.

Rectangles
~~~~~~~~~~
Rectangles with stripes of different intensities. Corresponding function - :code:`rectangles`.

Circles
~~~~~~~
Circles with inner circles of different intensities. Corresponding function - :code:`circles`.

