Labeling tool
=============
The goal of the labeling tool is to allow the user to align manually any two images and extract
the displacement field.

Launching
---------
To launch the labeling tool one needs to use the entry point :code:`label-tool`.

.. code-block:: bash

	label-tool --help
	usage: label-tool [-h] [-s] ref mov output_path

	positional arguments:
	  ref          Either a path to a reference image or a number from [0, 528)
		       representing the coronal dimension in the nissl stain volume.
	  mov          Path to a moving image. Needs to be of the same shape as
		       reference.
	  output_path  Folder where the outputs will be stored.

	optional arguments:
	  -h, --help   show this help message and exit
	  -s, --swap   Swap to the moving to reference mode. (default: False)

Arguments
~~~~~~~~~

1. :code:`ref` - number from 0 - 527 representing coronal section or a path to a custom reference image
2. :code:`mov` - path to the moving image that needs to have the same shape as the reference one
3. :code:`output_path` - path to folder where output is saved

Options
~~~~~~~

- :code:`swap` - flag that if activated the first landmark of a pair is on moving image otherwise it is on the reference



Interface
---------
.. image:: ../_images/labeling_tool.png
  :width: 600
  :alt: Inverse
  :align: center

The user adds landmark points until he/she is fully happy with the registration. Once the GUI is closed the
resulting displacement field is stored into the :code:`OUTPUT_PATH`.

1
~
The **interactive window** where the user specifies pairs of points that represent the landmarks in the
moving (resp. reference) image. The landmarks in the moving image are represented by **stars** and the ones in the
reference image by **circles**. To delete a point one needs to point at a reference point and then press a spacebar.

2
~
The **result window** that displays the current registered image and the reference. Note that one can also see
the deformation of a regular grid if one clicks on the **Show grid** button.

3
~
**Widgets** - color and thresholding sliders. Additionally, there are buttons:

- **Reset** - Deletes all registration point
- **Show Grid** - Shows regular grid warped by the current displacement. Red spots represent regions that are not invertible.
- ...

4
~
**Colormaps** for the reference and moving image.

5
~
**Interpolation window**. The first column represents the general algorithm

- **griddata** - Delaunay triangulation followed by affine interpolation for each triangle
- **rbf** - Interpolation using basis function

The second column is only used for the **rbf** interpolation and it specifies the kernel.

6
~
**Overview statistics**

- **Transformation quality** - If lower than 100% then displacement contains folds. Check the grid to see where exactly.
- **Average displacement** - Average displacement size


FAQ
---

- **Can the moving and reference image have different sizes?** No, make sure they are the same and one can resize the displacement field after registration
- **How do I save the displacement?** Simply by closing the GUI
- **Can I delete some landmark pairs?** Yes, point at the reference landmark and press spacebar
