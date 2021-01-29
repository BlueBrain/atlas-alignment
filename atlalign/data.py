"""A set of function generating simple datasets.

Notes
-----
All returned np.ndarrays should have dtype=np.float32 and intensities in range[0, 1] just to prevent scaling issues
withing ML models.
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

import json
import os
import warnings

import h5py
import numpy as np
from skimage.draw import circle
from skimage.util import img_as_float32

from atlalign.base import GLOBAL_CACHE_FOLDER

warnings.simplefilter("always", DeprecationWarning)


# DUMMY
def rectangles(n_samples, shape, height, width, n_levels=3, random_state=None):
    """Generate simple rectangles whose intensities gradually change.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    shape : tuple
        Represents the (height, width) of the output image (not the rectangle).

    height : int or tuple
        If int, then fixed size. If tuple than (height_min, heigh_max) and sampled uniformly.

    width : int or tuple
        If int, then fixed size. If tuple than (width_min, width_max) and sampled uniformly.

    n_levels : int or tuple, optional
        If int, then fixed levels. If tuples, then (n_levels_min, n_levels_max) and sampled uniformly.

    random_state : int, optional
        If int, then results are reproducible.

    Returns
    -------
    dataset : np.ndarray
        Of shape (n_samples, shape[0], shape[1], 1).

    """
    if not len(shape) == 2:
        raise ValueError(
            "The shape needs to have a length of 2. Current {}".format(len(shape))
        )

    h_img, w_img = shape

    if not (
        isinstance(height, (int, tuple))
        and isinstance(width, (int, tuple))
        and isinstance(n_levels, (int, tuple))
    ):
        raise TypeError(
            "Wrong type! height, width and n_levels need to be int or tuple"
        )

    height_min, height_max = (
        height if isinstance(height, tuple) else (height, height + 1)
    )
    width_min, width_max = width if isinstance(width, tuple) else (width, width + 1)
    n_levels_min, n_levels_max = (
        n_levels if isinstance(n_levels, tuple) else (n_levels, n_levels + 1)
    )

    if not (h_img >= height_max and w_img >= width_max):
        raise ValueError("The rectangle is too big!")

    if not (height_min > n_levels_max and width_min > n_levels_max):
        raise ValueError("Too many levels")

    dataset_list = []

    for _ in range(n_samples):

        img = np.zeros(shape, dtype="float32")

        if random_state is None:
            pass

        else:
            random_state += 1  # We want reproducible but different for each iteration
            np.random.seed(random_state)

        height_ = np.random.randint(height_min, height_max)
        width_ = np.random.randint(width_min, width_max)
        n_levels_ = np.random.randint(n_levels_min, n_levels_max)

        ul_r = np.random.randint(0, h_img - height_)
        ul_c = np.random.randint(0, w_img - width_)

        direction = np.random.choice(["up_down", "left_right"])
        # direction = 'up_down'

        if direction == "up_down":
            per_level = height_ // n_levels_

            r = ul_r
            intensity = 1 / n_levels_
            while r <= ul_r + height_ and intensity <= 1:
                img[r : r + per_level, ul_c : ul_c + width_] = intensity
                r += per_level
                intensity += 1 / n_levels_

        elif direction == "left_right":
            per_level = width_ // n_levels_

            c = ul_c
            intensity = 1 / n_levels_
            while c <= ul_c + width_ and intensity <= 1:
                img[ul_r : ul_r + height_, c : c + per_level] = intensity
                c += per_level
                intensity += 1 / n_levels_

        else:
            pass

        dataset_list.append(img[:, :, np.newaxis])

    return np.array(dataset_list, dtype="float32")


def circles(n_samples, shape, radius, n_levels=3, random_state=None):
    """Generate simple nested circles whose intensities gradually change.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    shape : tuple
        Represents the (height, width) of the output image (not the rectangle).

    radius : int or tuple
        If int, then all the outer circle always the same radius. If tuple, then represents
        (radius_min, radius_max) and the actual radius for a given sample is sampled from
        a uniform distribution.


    n_levels : int or tuple, optional
        If int, then fixed levels (nested circles). If tuples, then (n_levels_min, n_levels_max) and sampled uniformly.

    random_state : int, optional
        If int, then results are reproducible.

    Returns
    -------
    dataset : np.ndarray
        Of shape (n_samples, shape[0], shape[1], 1)

    """
    if not (isinstance(radius, (int, tuple)) and isinstance(n_levels, (int, tuple))):
        raise TypeError("The radius has a wrong type of {}".format(type(radius)))

    if not len(shape) == 2:
        raise ValueError(
            "The shape needs to have a length of 2. Current {}".format(len(shape))
        )

    height, width = shape

    dataset_list = []

    radius_min, radius_max = (
        radius if isinstance(radius, tuple) else (radius, radius + 1)
    )
    n_levels_min, n_levels_max = (
        n_levels if isinstance(n_levels, tuple) else (n_levels, n_levels + 1)
    )
    center_c_min, center_c_max = radius_max + 1, width - radius_max - 1
    center_r_min, center_r_max = radius_max + 1, height - radius_max - 1

    if not (2 * radius_max < width and 2 * radius_max < height):
        raise ValueError(
            "The radius is too high. Needs to be at max half of the min(height, width)"
        )

    for _ in range(n_samples):

        img = np.zeros(shape, dtype="float32")

        if random_state is None:
            pass

        else:
            random_state += 1  # We want reproducible but different for each iteration
            np.random.seed(random_state)

        c = np.random.randint(center_c_min, center_c_max)
        r = np.random.randint(center_r_min, center_r_max)
        n_levels_ = np.random.randint(n_levels_min, n_levels_max)
        outer_radius = np.random.randint(radius_min, radius_max)
        direction = np.random.choice(["incr", "decr"])  # From outer circle to center

        radi = np.linspace(
            outer_radius / n_levels_, outer_radius, n_levels_
        )  # Equally spaced
        intensity = 1 / n_levels_ if direction == "incr" else 1

        for rs in reversed(radi):  # Start from the largest circle
            res = circle(r, c, rs)
            img[res] = intensity

            if direction == "incr":
                intensity += 1 / n_levels_

            else:
                intensity -= 1 / n_levels_

        dataset_list.append(img[:, :, np.newaxis])

    return np.array(dataset_list, dtype="float32")


# NON DUMMY
def annotation_volume(path=None):
    """Output a dataset created of 528 consecutive coronal slice annotations.

    Notes
    -----
    As opposed to other datasets in this module the output ndim is 3 since we are not expecting to
    use this as a channel in an input.

    Parameters
    ----------
    path : str or None or LocalPath
        An absolute path to the underlying .npy file. If not speficied
        then a default one used.

    Returns
    -------
    x_atlas : np.ndarray
        An array of shape (528, 320, 456) representing the consecutive coronal slices. The dtype is np.int32 and
        the number represent distinct classes.

    """
    path = path or (GLOBAL_CACHE_FOLDER / "annotation.npy")

    atlas_volume = np.load(str(path)).astype(
        "int32"
    )  # saved as float but actually just integers

    return atlas_volume


def nissl_volume(path=None):
    """Output a dataset created of 528 consecutive coronal slices with Nissl staining.

    Parameters
    ----------
    path : str or None or LocalPath
        An absolute path to the underlying .npy file. If not speficied
        then a default one used.

    Returns
    -------
    x_atlas : np.ndarray
        An array of shape (528, 320, 456, 1) representing the consecutive coronal slices. The dtype is np.float32

    """
    path = path or (GLOBAL_CACHE_FOLDER / "nissl.npy")

    atlas_volume = np.load(str(path)).astype(
        "uint8"
    )  # saved as float but actually just integers
    atlas_volume_float = np.array(
        [img_as_float32(slc) for slc in atlas_volume]
    )  # deals with scaling too!

    return atlas_volume_float[:, :, :, np.newaxis]


def manual_registration(path=None):
    """Return all manual registration done with the new labeling tool.

    Parameters
    ----------
    path : str or None or LocalPath
        An absolute path to the underlying .h5 file. If not speficied
        then a default one used.

    Returns
    -------
    res : dict
        Dictionary where keys are corresponding dataset names. The values are numpy arrays.
    """
    path = path or (GLOBAL_CACHE_FOLDER / "manual_registration.h5")

    with h5py.File(str(path), "r") as f:
        return {k: f[k][:] for k in f.keys()}


def _get_all_ids(folder):
    """Return a set of all integers that occur at a beginning of a file name.

    Notes
    -----
    We want to extract all image_ids withing a section dataset folder. Used within `csaba_registration`.

    Parameters
    ----------
    folder : str
        Absolute path to a folder.

    Returns
    -------
    res : set
        Set of all image_ids. We use ``set`` in order to avoid duplicates.

    """
    res = set()
    for _, _, files in os.walk(folder):
        for f in files:
            maybe = f.split("_")[0]
            try:
                res.add(int(maybe))
            except ValueError:
                # Impossible to convert to int - the string doesnt look like 'NUMBER_whatever'
                pass
        break

    return res


def _get_section_numbers(folder, id_list):
    """For a list of section image ids find section numbers.

    Notes
    -----
    Used within `csaba_registration`. Note that these section numbers are manually read from text files. Allen's
    API is not used.

    Parameters
    ----------
    folder : str
        Absolute path of a folder.

    id_list : list
        List of image ids.

    Returns
    -------
    res : dict
        The keys are image_ids and the values are section numbers.

    """
    res = {}
    for id_ in id_list:
        file_path = folder + "{}_section_id.txt".format(id_)

        with open(file_path) as f:
            try:
                res[id_] = int(f.read())

            except Exception:
                pass

    return res


def segmentation_collapsing_labels(path=None):
    """Segmentation collapsing tree.

    Parameters
    ----------
    path : str or None or LocalPath
        An absolute path to the underlying .json file. If not speficied
        then a default one used.

    Returns
    -------
    json_file : dict
        Dictionary containing all the labels in a tree structure.

    """
    path = path or (GLOBAL_CACHE_FOLDER / "annotation_hierarchy.json")

    with open(str(path), "r") as json_file:
        json_dict = json.load(json_file)

    return json_dict
