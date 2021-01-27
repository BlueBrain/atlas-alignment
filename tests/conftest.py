"""Define all fixtures.

Notes
-----
A lot of the fixtures are built on top of the following 4 atomic fixtures:
    * img_grayscale_uint
    * img_grayscale_float
    * img_rgb_uint
    * img_rgb_float

The fixtures below are just parameterized in a way that only a subset of the above
atomic fixtures is used
    * img
    * img_grayscale
    * img_rgb
    * img_uint
    * img_float
"""

"""
    The package atlalign is a tool for registration of 2D images.

    Copyright (C) 2021 EPFL/Blue Brain Project

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pathlib

import cv2
import numpy as np
import pytest
from skimage.util import img_as_float32

from atlalign.base import DisplacementField

SHAPE = (20, 30)  # used for `img_dummy` and `img_random`
RANDOM_STATE = 2

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def path_test_data():
    return ROOT_PATH / "tests" / "data"


@pytest.fixture(scope="session")
def img_grayscale_uint(path_test_data):
    """Generate a grayscale image with dtype uint8."""
    file_path = path_test_data / "animals.jpg"

    img_out = cv2.imread(str(file_path), 0)

    assert img_out.ndim == 2
    assert img_out.dtype == np.uint8
    assert np.all(img_out >= 0) and np.all(img_out <= 255)

    return img_out


@pytest.fixture()
def img_grayscale_float(img_grayscale_uint):
    """Generate a float32 version of the grayscale image."""
    img_out = img_as_float32(img_grayscale_uint, force_copy=True)

    assert img_out.ndim == 2
    assert img_out.dtype == np.float32
    assert np.all(img_out >= 0) and np.all(img_out <= 1)

    return img_out


@pytest.fixture(scope="session")
def img_rgb_uint(path_test_data):
    """Just a RGB image with dtype uint8.

    Notes
    -----
    OpenCV reads a 3 channel image as 'BGR' but it is not a problem for our purposes.

    """
    file_path = path_test_data / "animals.jpg"

    img_out = cv2.imread(str(file_path), 1)

    assert img_out.ndim == 3
    assert img_out.dtype == np.uint8
    assert np.all(img_out >= 0) and np.all(img_out <= 255)

    return img_out


@pytest.fixture()
def img_rgb_float(img_rgb_uint):
    """Generate a float32 version of the rgb image."""
    img_out = img_as_float32(img_rgb_uint, force_copy=True)

    assert img_out.ndim == 3
    assert img_out.dtype == np.float32
    assert np.all(img_out >= 0) and np.all(img_out <= 1)

    return img_out


@pytest.fixture(
    params=["grayscale_uint8", "grayscale_float32", "RGB_uint8", "RGB_float32"]
)
def img(request, img_rgb_uint, img_grayscale_uint, img_rgb_float, img_grayscale_float):
    """Generate parametrized fixture capturing all 4 possible uint8/float32 and grayscale/rgb combinations.

    Notes
    -----
    If this fixture used then the test will run automatically on all 4 of these.

    """
    img_type = request.param

    if img_type == "grayscale_uint8":
        return img_grayscale_uint

    elif img_type == "grayscale_float32":
        return img_grayscale_float

    elif img_type == "RGB_uint8":
        return img_rgb_uint

    elif img_type == "RGB_float32":
        return img_rgb_float

    else:
        raise ValueError("Unrecognized image type {}".format(img_type))


@pytest.fixture(params=["grayscale_float32", "RGB_float32"])
def img_float(request, img_rgb_float, img_grayscale_float):
    """Generate a parametrized fixture capturing all 2 possible float32 -> grayscale and rgb.

    Notes
    -----
    If this fixture used then the test will run automatically on all 2 of these.

    """
    img_type = request.param

    if img_type == "grayscale_float32":
        return img_grayscale_float

    elif img_type == "RGB_float32":
        return img_rgb_float

    else:
        raise ValueError("Unrecognized image type {}".format(img_type))


@pytest.fixture(params=["grayscale_uint8", "RGB_uint8"])
def img_uint(request, img_rgb_uint, img_grayscale_uint):
    """Generate a parametrized fixture capturing all 2 possible uint8 -> grayscale and rgb.

    Notes
    -----
    If this fixture used then the test will run automatically on all 2 of these.

    """
    img_type = request.param

    if img_type == "grayscale_uint8":
        return img_grayscale_uint

    elif img_type == "RGB_uint8":
        return img_rgb_uint

    else:
        raise ValueError("Unrecognized image type {}".format(img_type))


@pytest.fixture(params=["grayscale_uint8", "grayscale_float32"])
def img_grayscale(request, img_grayscale_uint, img_grayscale_float):
    """Generate a parametrized fixture capturing all 2 possible grayscale -> uint8 and float32.

    Notes
    -----
    If this fixture used then the test will run automatically on all 2 of these.

    """
    img_type = request.param

    if img_type == "grayscale_uint8":
        return img_grayscale_uint

    elif img_type == "grayscale_float32":
        return img_grayscale_float

    else:
        raise ValueError("Unrecognized image type {}".format(img_type))


@pytest.fixture(params=["RGB_uint8", "RGB_float32"])
def img_rgb(request, img_rgb_uint, img_rgb_float):
    """Generate a parametrized fixture capturing all 2 possible rgb  -> uint8 and float32.

    Notes
    -----
    If this fixture used then the test will run automatically on all 2 of these.

    """
    img_type = request.param

    if img_type == "RGB_uint8":
        return img_rgb_uint

    elif img_type == "RGB_float32":
        return img_rgb_float

    else:
        raise ValueError("Unrecognized image type {}".format(img_type))


@pytest.fixture(scope="function")  # In order to allow for in place changes
def img_dummy():
    """Generate a dummy image made out of all zeros."""
    return np.zeros(SHAPE, dtype=np.float32)


@pytest.fixture(scope="function")
def img_random():
    """Generate a dummy images made out of random intensities."""
    np.random.seed(RANDOM_STATE)
    out = np.random.random(SHAPE).astype(dtype=np.float32)

    assert out.dtype == np.float32

    return out


@pytest.fixture(scope="session")
def df_cached(path_test_data):
    """Load a DVF and its inverse of shape (80, 114).

    DVF represents a mild warping then can be relatively easily unwarped.

    Notes
    -----
    After composition the largest displacement in x ~ 0.05 and in y ~ 0.03.

    Returns
    -------
    delta_x : np.array
        DVF in the x direction.

    delta_y : np.array
        DVF in the y direction.

    delta_x_inv : np.array
        Inverse DVF in the x direction.

    delta_x_inv : np.array
        Inverse DVF in the y direction.

    """
    file_path = path_test_data / "mild_inversion.npy"
    a = np.load(str(file_path))

    delta_x = a[..., 0]
    delta_y = a[..., 1]
    delta_x_inv = a[..., 2]
    delta_y_inv = a[..., 3]

    return delta_x, delta_y, delta_x_inv, delta_y_inv


@pytest.fixture()
def df_id(request):
    """Generate an identity transformation.

    In order to specify a shape one decorates the test function in the following way:
     `@pytest.mark.parametrize('df_id', [(320, 456)], indirect=True)`

    """
    if hasattr(request, "param"):
        shape = request.param
    else:
        shape = (10, 11)
    return DisplacementField.generate(shape, approach="identity")


@pytest.fixture()
def label_dict():
    """Gnerate a dictionary for the concatenation of labels (segmentation)."""
    dic = {
        "id": 2,
        "children": [
            {"id": 3, "children": []},
            {
                "id": 4,
                "children": [{"id": 5, "children": []}, {"id": 6, "children": []}],
            },
        ],
    }

    return dic
