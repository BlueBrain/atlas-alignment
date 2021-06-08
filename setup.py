from setuptools import find_packages, setup

import atlalign

# install with `pip install -e .`

# Preparations
VERSION = atlalign.__version__
DESCRIPTION = "Blue Brain multi-modal registration and alignment toolbox"

LONG_DESCRIPTION = """
Atlas Alignment is a toolbox to perform multimodal image registration. It
includes both traditional and supervised deep learning models.

This project originated from the Blue Brain Project efforts on aligning mouse
brain atlases obtained with ISH gene expression and Nissl stains."""

PYTHON_REQUIRES = ">=3.6.0"
INSTALL_REQUIRES = [
    "antspyx==0.2.7",
    "imgaug<0.3",
    "matplotlib>=3.0.3",
    "mlflow",
    "nibabel>=2.4.0",
    "numpy",
    "seaborn",
    "scikit-image>=0.16.0",
    "scikit-learn>=0.20.2",
    "scipy",
    "tensorflow>=2.4.0",
    "tensorflow-addons",  # For resampler in atlalign/ml_utils/layers.py
]

setup(
    name="atlalign",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/BlueBrain/atlas_alignment",
    author="Blue Brain Project, EPFL",
    license="LGPLv3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "black>=20.8b1",
            "flake8>=3.7.4",
            "pydocstyle>=3.0.0",
            "pytest>=3.10.1",
            "pytest-cov",
            "pytest-mock>=1.10.1",
        ],
        "docs": ["sphinx>=1.3", "sphinx-bluebrain-theme"],
    },
    entry_points={"console_scripts": ["label-tool = atlalign.label.cli:main"]},
)
