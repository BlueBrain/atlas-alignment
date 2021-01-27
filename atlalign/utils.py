"""Collection of helper classes and function that do not deserve to be in base.py.

Notes
-----
This module cannot import from anywhere else within this project to prevent circular dependencies.

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

import numpy as np
import scipy.spatial.qhull as qhull


def _triangulate(xyz, uvw):
    """Perform Delaunay triangulation.

    Parameters
    ----------
    xyz : np.ndarray
        An array of shape (N, 2) where each row represents one point in 2D (stable points) for which we know the
        function value.

    uvw : np.ndarray
        An array of shape (K, 2) where each row represents one point in 2D (query point) for which we want to
        interpolate the function value.

    Returns
    -------
    vertices : np.ndarray
        An array of shape (K, 3) representing the triangle vertices of each query point. Note that these
        vertices are always stable points (from range [O, N))

    wts : np.ndarray
        An array of shape (K, 3) representing the weights of respective vertices at each query point.

    """
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, 2]
    bary = np.einsum("njk,nk->nj", temp[:, :2, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    return vertices, wts


def _interpolate(values, vertices, wts):
    """Interpolate inside a triangle.

    Parameters
    ----------
    values : np.ndarray
        An array of shape (N,) that represents function value on the know points which are the vertices after
        Delaunay triangulation.

    vertices : np.ndarray
        An array of shape (K, 3) representing the triangle vertices of each query point. Note that these
        vertices are always stable points (from range [O, N)).

    wts : np.ndarray
        An array of shape (K, 3) representing the weights of respective vertices at each query point.

    Returns
    -------
    interpolations : np.ndarray
        An array of shape (K,) representing the interpolated function values on the query points.

    """
    return np.einsum("nj,nj->n", np.take(values, vertices), wts)


def griddata_custom(points, values_f_1, values_f_2, xi):
    """Run griddata extensions that performs only one triangulation.

    Notes
    -----
    The scipy implementation does not allow to separate triangulation from interpolation. Since we need
    to evaluate 2 different functions on the !same! non-regular grid if points the triangulation can be simply
    just done once and stored.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (N, 2) where each row represents one point in 2D for which we know the function value.

    values_f_1 : np.ndarray
        An array of shape (N,) where each row represents a value of function f_1 on the corresponding point in `points`.

    values_f_2 : np.ndarray
        An array of shape (N,) where each row represents a value of function f_2 on the corresponding point in `points`.

    xi : tuple
        Tuple of 2 np.ndarray of shapes (h, w) representing the x and y coordinates of the points where we want to
        interpolate data. Note that this is simply the result of `np.meshgrid` if our points of interest lie on a
        regular grid.

    Returns
    -------
    f_1_interpolation_on_xi : np.ndarray
        An array of shape (h, w) representing the interpolation of f_1 on the `xi` points.

    f_2_interpolation_on_xi : np.ndarray
        An array of shape (h, w) representing the interpolation of f_2 on the `xi` points.

    References
    ----------
    https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids  # noqa

    """
    if isinstance(xi, tuple):
        shape = xi[0].shape
        xi = np.hstack((xi[0].reshape(-1, 1), xi[1].reshape(-1, 1)))  # possible speedup

    else:
        raise TypeError("The xi needs to be a tuple of equally shaped np.ndarrays.")

    vertices, wts = _triangulate(points, xi)

    f_1_interpolation_on_xi = _interpolate(values_f_1, vertices, wts).reshape(shape)
    f_2_interpolation_on_xi = _interpolate(values_f_2, vertices, wts).reshape(shape)

    return f_1_interpolation_on_xi, f_2_interpolation_on_xi


def _find_all_children(d, children_list=None):
    """Construct a list of all the ids of the children of a node and the node itself.

    Parameters
    ----------
    d : dict
        Dictionary node from whom we want the list of all the children and children's children.

    children_list : list, default None
        List of children which has to be empty for the first iteration of the function.

    Returns
    -------
    children_list : list
        List of children's ids.

    """
    if children_list is None:
        children_list = []

    for key, value in d.items():
        if key == "id":
            children_list.append(value)
        if isinstance(value, list):
            for child in value:
                _find_all_children(child, children_list)

    return children_list


def _find_concatenate_labels(d, chosen_depth, dict_of_labels=None, current_depth=0):
    """Construct a dictionary which has for each key, the value of the new label after concatenation.

    Parameters
    ----------
    d : dict
        Dictionary node for which we want to concatenate some ids depending on the depth branch.

    chosen_depth : int
        Depth at which it is wanted to concatenate the labels.

    dict_of_labels : dict, default {}
        Dictionary of corresponding labels (empty at the first call).

    current_depth : int, default 0
        Depth of the dictionary node.

    Returns
    -------
    dict_of_labels: dict
        Dictionary of corresponding labels after concatenation of labels tree.

    """
    if dict_of_labels is None:
        dict_of_labels = {}

    if current_depth < chosen_depth:
        for key, value in d.items():
            if key == "id":
                dict_of_labels[value] = value
            if isinstance(value, list):
                current_depth = current_depth + 1
                for child in value:
                    _find_concatenate_labels(
                        child,
                        chosen_depth,
                        dict_of_labels=dict_of_labels,
                        current_depth=current_depth,
                    )
    else:
        children_list = []
        _find_all_children(d, children_list)
        for key, value in d.items():
            if key == "id":
                for child in children_list:
                    dict_of_labels[child] = value

    return dict_of_labels


def find_labels_dic(segmentation_array, dic, chosen_depth):
    """Collapse existing labels into parent labels corresponding to the tree provided in a dictionary.

    Parameters
    ----------
    segmentation_array : np.array
        Annotation array before the concatenation of the labels.

    dic : dict
        Dictionary of tree of labels.

    chosen_depth : int
        Depth at which it is wanted to concatenate the labels.

    Returns
    -------
    new_segmentation_array : np.array
        New Annotation array with the concatenation of the labels at the desired depth. If a specific label
        does not exist in the tree it is assigned -1.

    """
    labels_dic = _find_concatenate_labels(dic, chosen_depth)

    new_segmentation_array = segmentation_array.copy()
    all_labels = np.unique(segmentation_array)

    for label in all_labels:
        if label != 0:
            new_label = labels_dic.get(label, -1)
            new_segmentation_array[new_segmentation_array == label] = new_label

    return new_segmentation_array
