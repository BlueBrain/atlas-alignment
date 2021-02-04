.. _installation:

Installation
============
It is highly recommended to install the project into a new virtual environment.

Python Requirements
-------------------
The project is only available for **Python 3.7**. The main reason for this 
restriction is an external dependency **ANTsPy** that does
not provide many precompiled wheels on PyPI.

External Dependencies
---------------------
Some of the functionalities of :code:`atlalign` depend on the
`TensorFlow implementation of the Learned Perceptual Image Patch Similarity <https://github.com/alexlee-gk/lpips-tensorflow>`_.
Unfortunately, the
package is not available on PyPI and must be installed manually as follows.

.. code-block:: bash

    pip install git+http://github.com/alexlee-gk/lpips-tensorflow.git#egg=lpips_tf

You can now move on to installing the actual `atlalign` package!

Installation from PyPI
----------------------
The :code:`atlalign` package can be easily installed from PyPI.

.. code-block:: bash

    pip install atlalign

Installation from source
------------------------
As an alternative to installing from PyPI, if you want to try the latest version
you can also install from source.

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/atlas_alignment#egg=atlalign

Development installation
------------------------
For development installation one needs additional dependencies grouped in :code:`extras_requires` in the
following way:

- **dev** - pytest + plugins, flake8, pydocstyle, tox
- **docs** - sphinx

.. code-block:: bash

    git clone https://github.com/BlueBrain/atlas_alignment
    cd atlas_alignment
    pip install -e .[dev,docs]


Generating documentation
------------------------
To generate the documentation make sure you have dependencies from :code:`extras_requires` - :code:`docs`.

.. code-block:: bash

    cd docs
    make clean && make html

One can view the docs by opening :code:`docs/_build/html/index.html` in a browser.
