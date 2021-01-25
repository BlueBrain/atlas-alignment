.. _installation:

Installation
============
It is highly recommended to install the project into a new virtual environment.

Requirements
------------
The project is only available for **Python 3.7**. The main reason for this 
restriction is an external dependency **ANTsPy** that does
not provide many precompiled wheels on PyPI.


Standard installation
---------------------
First of all make sure the :code:`pip` version is at least :code:`19.1`
(`PyPA <https://pip.pypa.io/en/stable/reference/pip_install/#requirement-specifiers>`_).

.. code-block:: bash

    pip install --upgrade pip
    pip --version  # make sure >= 19.1

Then one can move on to installing the actual :code:`atlalign` package:

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/atlas_alignment#egg=atlalign[tf]

The extras entry :code:`[tf]` represents **TensorFlow** and can be dropped if it is already installed.


Development installation
------------------------
For development installation one needs additional dependencies grouped in :code:`extras_requires` in the
following way:

- **dev** - pytest + plugins, flake8, pydocstyle, tox
- **docs** - sphinx
- **tf** - tensorflow

.. code-block:: bash

    git clone https://github.com/BlueBrain/atlas_alignment
    cd atlas_alignment
    pip install -e .[dev,docs,tf]


Generating documentation
------------------------
To generate the documentation make sure you have dependencies from :code:`extras_requires` - :code:`docs`.

.. code-block:: bash

    cd docs
    make clean && make html

One can view the docs by opening :code:`docs/_build/html/index.html` in a browser.
