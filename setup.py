from setuptools import setup, find_packages

import atlalign

# install with `pip install -e .`

# Preparations
VERSION = atlalign.__version__

PYTHON_REQUIRES = "~=3.7.0"
INSTALL_REQUIRES = [
    "antspyx==0.2.4",
    "allensdk>=0.16.3",
    "imgaug<0.3",
    "Keras==2.2.4",
    "keras_contrib @ git+http://github.com/keras-team/keras-contrib.git@e1574a1#egg=keras_contrib",
    "lpips_tf @ git+http://github.com/alexlee-gk/lpips-tensorflow.git#egg=lpips_tf",
    "matplotlib>=3.0.3",
    "mlflow",
    "nibabel>=2.4.0",
    "numpy>=1.16.3",
    "statsmodels>=0.9.0",
    "scikit-image>=0.15.0",
    "scikit-learn>=0.20.2",
    "scipy==1.2.1",
]

setup(
    name="atlalign",
    version=VERSION,
    description="Image registration with deep learning",
    author="Blue Brain Project, EPFL",
    packages=find_packages(),
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "black>=20.8b1",
            "flake8>=3.7.4",
            "pydocstyle>=3.0.0",
            "pytest>=3.10.1",
            "pytest-benchmark>=3.2.2",
            "pytest-cov",
            "pytest-mock>=1.10.1",
        ],
        "docs": ["sphinx>=1.3", "sphinx-bluebrain-theme"],
        "tf": ["tensorflow>=1.15.4,<2"],
    },
    entry_points={"console_scripts": ["label-tool = atlalign.label.cli:main"]},
)
