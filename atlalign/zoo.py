"""Contains different way how to generate displacement field.

Notes
-----
Ideally you want the only positional argument to be the shape and all the others be keyword arguments with
reasonable defaults.
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

import math

import cv2
import numpy as np
from scipy.interpolate import Rbf, SmoothBivariateSpline, griddata
from skimage.filters import gaussian
from skimage.transform import AffineTransform, ProjectiveTransform, SimilarityTransform

from atlalign.utils import griddata_custom


def affine(shape, matrix=None):
    """Affine transformation encoded in a 2 x 3 matrix.

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).

    matrix : np.ndarray
        Transformation matrix of the shape 2 x 3.

    Raises
    ------
    ValueError
        In case the transformation matrix has a wrong shape.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates.

    """
    if matrix is None:
        matrix = np.eye(3)

    if matrix.shape == (2, 3):
        matrix = np.vstack(
            (matrix, [0, 0, 1])
        )  # just add the homogeneous coordinates parts

    if matrix.shape != (3, 3):
        raise ValueError(
            "The shape of affine transformation matrix is {}, correct is (3, 3)".format(
                matrix.shape
            )
        )

    tform = AffineTransform(matrix)

    x, y = np.meshgrid(range(shape[1]), range(shape[0]))
    coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))).astype(float)

    coords_after = tform(coords)
    coords_delta = coords_after - coords

    delta_x = np.reshape(coords_delta[:, 0], shape)
    delta_y = np.reshape(coords_delta[:, 1], shape)

    return delta_x, delta_y


def affine_simple(
    shape,
    scale_x=1,
    scale_y=1,
    rotation=0,
    translation_x=0,
    translation_y=0,
    shear=0,
    apply_centering=True,
):
    """Just a human version of affine mapping.

    Notes
    -----
    Instead of specifying the whole matrix one can just specify all the understandable quantities.

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).

    scale_x : float
        Scale on the x axis. If scale_x < 1 then zoom out, if scale_x > 1 zoom in.

    scale_y : float
        Scale on the y axis. If scale_y < 1 then zoom out, if scale_y > 1 zoom in.

    rotation : float
          Rotation angle in counter-clockwise direction as radians.

    translation_x : float
        Translation in the x direction. If translation_x > 0 then to the right, else to the left.

    translation_y : float
        Translation in the y direction. If translation_y > 0 then down, else to the up.

    shear : float
        Shear angle in counter-clockwise direction as radians.

    apply_centering : bool
        If True then (h // 2 - 0.5, w // 2 - 0.5) is considered a center of the image. And before performing all the
        other operations the image is first shifted so that the center corresponds to (0, 0). Then the actual
        transformation is applied and after that the image is shifted into the original center.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates.

    """
    matrix = np.array(
        [
            [
                scale_x * math.cos(rotation),
                -scale_y * math.sin(rotation + shear),
                translation_x,
            ],
            [
                scale_x * math.sin(rotation),
                scale_y * math.cos(rotation + shear),
                translation_y,
            ],
            [0, 0, 1],
        ]
    )

    if not apply_centering:
        return affine(shape, matrix)

    center_rc = np.array([(shape[0] / 2) - 0.5, (shape[1] / 2) - 0.5])
    center_xy = np.array([center_rc[1], center_rc[0]])

    tform1 = SimilarityTransform(translation=center_xy)
    tform2 = SimilarityTransform(matrix=matrix)
    tform3 = SimilarityTransform(translation=-center_xy)
    tform = tform3 + tform2 + tform1

    return affine(shape, tform.params)


def control_points(
    shape,
    points=None,
    values_delta_x=None,
    values_delta_y=None,
    anchor_corners=True,
    interpolation_method="griddata",
    interpolator_kwargs=None,
):
    """Simply interpolate given control points.

    Notes
    -----
    We assume there are N control points.

    This function is used by others functions from the `zoo`. See below a complete list
        - `edge_stretching`
        - `single_frequency`

    Additionally, it is also used in the `eliminate_bb` method of the ``DisplacementField`` class.


    Parameters
    ----------
    shape : tuple
        Of the form (height, width).

    points : np.ndarray, optional
        An array of shape (N, 2) where each row represents a (row, column) of a given control point.

    values_delta_x : np.ndarray, optional
        An array of shape (N, ) where each row represents a delta_x of the transformation at the corresponding
        control point.

    values_delta_y : np.ndarray, optional
        An array of shape (N, ) where each row represents a delta_y of the transformation at the corresponding
        control point.

    anchor_corners : bool, optional
        If True then each of the 4 images corners are automatically added to the control points
        and identity transformation is assumed.

    interpolation_method : {'griddata', 'bspline', 'rbf'}, optional
        Interpolation method to use.

    interpolator_kwargs : dict, optional
        Additional parameters passed to the interpolator.


    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates of shape = `shape`.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates of shape = `shape`.

    Raises
    ------
    TypeError:
        When either of the `points`, `values_delta_x` or `values_delta_y` is not a np.ndarray.

    ValueError:
        Various inconsistencies in inputs (different len, zero len, wrong other dimensions or default without anchor).

    IndexError:
        Some of the points are outside of the image domain.

    """
    height, width = shape
    interpolator_kwargs = interpolator_kwargs or {}

    in_default_mode = (
        points is None and values_delta_x is None and values_delta_y is None
    )

    # Create a default
    if in_default_mode:
        if anchor_corners:
            points = np.array(
                [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
            )  # clockwise
            values_delta_x = np.zeros(4)
            values_delta_y = np.zeros(4)

        else:
            raise ValueError("Cannot instantiate")

    # Checks
    if not (
        isinstance(points, np.ndarray)
        and isinstance(values_delta_x, np.ndarray)
        and isinstance(values_delta_y, np.ndarray)
    ):
        raise TypeError("All inputs need to be a np.ndarray")

    if not (len(points) == len(values_delta_x) == len(values_delta_y)):
        raise ValueError("Arrays have different lengths")

    if len(points) == 0:
        raise ValueError("No points given.")

    N = len(points)

    if not (points.ndim == 2 and values_delta_x.ndim == 1 and values_delta_y.ndim == 1):
        raise ValueError("Wrong dimensions of input data")

    if not np.all([0 <= p[0] < height and 0 <= p[1] < width for p in points]):
        raise IndexError("Some control points are outside of image domain.")

    if not in_default_mode and anchor_corners:
        points_anchors = np.array(
            [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
        )  # clockwise
        values_delta_x_anchors = np.zeros((4, 1))
        values_delta_y_anchors = np.zeros((4, 1))

        points = np.vstack((points_anchors, points))
        values_delta_x = np.vstack(
            (values_delta_x_anchors, values_delta_x.reshape((N, 1)))
        )
        values_delta_y = np.vstack(
            (values_delta_y_anchors, values_delta_y.reshape((N, 1)))
        )

    x, y = np.meshgrid(list(range(width)), list(range(height)))
    xi = (y, x)

    if interpolation_method == "griddata":
        values_grid_delta_x = griddata(
            points=points, values=values_delta_x, xi=xi, **interpolator_kwargs
        )
        values_grid_delta_y = griddata(
            points=points, values=values_delta_y, xi=xi, **interpolator_kwargs
        )

        delta_x = values_grid_delta_x.reshape(shape)
        delta_y = values_grid_delta_y.reshape(shape)

    elif interpolation_method == "griddata_custom":
        # triangulation performed only once
        values_grid_x, values_grid_y = griddata_custom(
            points, values_delta_x, values_delta_y, xi
        )

        delta_x = values_grid_x.reshape(shape)
        delta_y = values_grid_y.reshape(shape)

    elif interpolation_method == "bspline":
        x_, y_ = points[:, 1], points[:, 0]

        ip_delta_x = SmoothBivariateSpline(
            x_, y_, values_delta_x, **interpolator_kwargs
        )
        ip_delta_y = SmoothBivariateSpline(
            x_, y_, values_delta_y, **interpolator_kwargs
        )

        delta_x = ip_delta_x(x.ravel(), y.ravel(), grid=False).reshape(shape)
        delta_y = ip_delta_y(x.ravel(), y.ravel(), grid=False).reshape(shape)

    elif interpolation_method == "rbf":
        x_, y_ = points[:, 1], points[:, 0]

        ip_delta_x = Rbf(x_, y_, values_delta_x, **interpolator_kwargs)
        ip_delta_y = Rbf(x_, y_, values_delta_y, **interpolator_kwargs)

        delta_x = ip_delta_x(x.ravel(), y.ravel()).reshape(shape)
        delta_y = ip_delta_y(x.ravel(), y.ravel()).reshape(shape)

    else:
        raise ValueError(
            "Unrecognized interpolation method {}".format(interpolation_method)
        )

    return delta_x, delta_y


def edge_stretching(
    shape,
    edge_mask=None,
    n_perturbation_points=3,
    radius_max=30,
    interpolation_method="griddata_custom",
    interpolator_kwargs=None,
):
    """Pick points on the edges and using them to stretch the image.

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).

    edge_mask : np.ndarray
        An array of dtype=bool of shape `shape`. The True elements represent an edge.

    n_perturbation_points : int
        Number of points to pick among the edges on which the perturbation defined.

    radius_max : float, optional
        Maxim value of radius, the actual value is a sample from uniform [0, radius_max].

    interpolation_method : {'griddata', 'griddata_custom', 'bspline', 'rbf'}, optional
        Interpolation method to use.

    interpolator_kwargs : dict, optional
        Additional parameters passed to the interpolator.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates.

    """
    if edge_mask is None:
        edge_mask = np.zeros(shape, dtype=bool)  # no edges

    if shape != edge_mask.shape:
        raise ValueError("The edge_mask has an incorrect shape.")

    n_edge_pixels = edge_mask.sum()
    n_perturbation_points = min(n_perturbation_points, n_edge_pixels)

    if n_perturbation_points == 0:
        return np.zeros(shape), np.zeros(
            shape
        )  # no perturbation points -> identity mapping

    edge_points = np.argwhere(edge_mask)

    ixs = np.random.choice(n_edge_pixels, replace=False, size=n_perturbation_points)
    points = edge_points[ixs]

    perturbation_radii = np.random.uniform(0, radius_max, size=n_perturbation_points)
    perturbation_angles = np.random.uniform(
        0, 2 * np.pi, size=n_perturbation_points
    )  # in radians

    pixel_displacements_rc = np.array(
        [
            np.array([-np.sin(angle), np.cos(angle)]) * radius
            for radius, angle in zip(perturbation_radii, perturbation_angles)
        ]
    )

    return control_points(
        shape=shape,
        points=points,
        values_delta_x=pixel_displacements_rc[:, 0],
        values_delta_y=pixel_displacements_rc[:, 1],
        interpolation_method=interpolation_method,
        interpolator_kwargs=interpolator_kwargs,
    )


def paper(
    shape,
    n_pixels=10,
    v_min=-20000,
    v_max=20000,
    kernel_sigma=25,
    p=None,
    random_state=None,
):
    """Algorithm proposed in the reference paper.

    Notes
    -----
    This algorithm has 2 steps
        1) Pick `n_pixels` in the image and randomly sample x and y displacement from interval [`v_min`, `v_max`]
        2) Apply a gaussian kernel on the displacement field with a given `kernel_size` and `kernel_sigma`

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).

    n_pixels : int
        Number of pixels to choose in the first step.

    v_min : float
        Minimum value for x and y displacement sampling.

    v_max : float
        Maximum value for x and y displacement sampling.

    kernel_sigma : float
        Standard deviation of the kernel in both the x and y direction.

    p : np.array
        Pixelwise probability of selection, where p.shape=shape. Note that if None, then uniform.

    random_state : int
        If None, then results not reproducible.


    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates.


    References
    ----------
    [1] Sokooti H., de Vos B., Berendsen F., Lelieveldt B.P.F., Išgum I., Staring M. (2017) Nonrigid Image
        Registration Using Multi-scale 3D Convolutional Neural Network

    """
    # Checks
    if n_pixels <= 0:
        raise ValueError(
            "The n_pixels needs to be higher than 0, current value is {}".format(
                n_pixels
            )
        )

    height, width = shape
    delta_x = np.zeros(
        shape, dtype=np.float32
    )  # makes GaussianBlur way quicker than float64
    delta_y = np.zeros(
        shape, dtype=np.float32
    )  # makes GaussianBlur way quicker than float64

    # Step 1
    np.random.seed(random_state)  # set seed
    pixel_count = np.prod(shape)
    if p is not None:
        p_ = p.ravel()
    else:
        p_ = None

    selected_pixels = np.random.choice(pixel_count, n_pixels, replace=False, p=p_)
    selected_pixels_rc = np.array([(x // width, x % width) for x in selected_pixels])
    xy_deltas = np.random.uniform(v_min, v_max, size=(n_pixels, 2))

    delta_y[selected_pixels_rc[:, 0], selected_pixels_rc[:, 1]] = xy_deltas[:, 1]
    delta_x[selected_pixels_rc[:, 0], selected_pixels_rc[:, 1]] = xy_deltas[:, 0]

    # Step 2
    delta_x = cv2.GaussianBlur(
        delta_x, ksize=(0, 0), sigmaX=kernel_sigma, sigmaY=kernel_sigma
    )
    delta_y = cv2.GaussianBlur(
        delta_y, ksize=(0, 0), sigmaX=kernel_sigma, sigmaY=kernel_sigma
    )

    return delta_x, delta_y


def paper_microsoft(shape, alpha=1000, sigma=10, random_state=None):
    """Generate artificial displacement based on a paper from Microsoft.

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).

    alpha : float
        Constant that the per pixel displacements are multiplied with. The higher the crazier displacements. If
        set to 0 then zero transformation.

    sigma : float
        Standard deviation of the gaussian kernel. The closer to 0 the crazier displacement. If close to inf then
        zero transformation.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates.

    References
    ----------
    Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003, August). Best practices for convolutional neural networks
    applied to visual document analysis. In null (p. 958). IEEE.

    """
    np.random.seed(random_state)
    random_delta_x_, random_delta_y_ = np.random.uniform(-1, 1, size=(2, *shape))
    delta_x = gaussian(random_delta_x_, sigma=sigma, cval=0) * alpha
    delta_y = gaussian(random_delta_y_, sigma=sigma, cval=0) * alpha

    return delta_x, delta_y


def patch_shift(
    shape, ul=(10, 10), height=100, width=120, shift_size=30, shift_direction="D"
):
    """For a fixed patch in an image redefine it with another same-shaped patch elsewhere in the image.

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).
    ul : tuple
        Of the form (row of the UPPER LEFT corner of the patch, column of the UPPER LEFT corner of the patch).

    height : int
        Height of the patch.

    width : int
        Width of the patch.

    shift_size : int
        How many pixels to shift the patch.

    shift_direction : str, {'U', 'D', 'L', 'R'}
        The direction of the shift. 'U' = Up, 'D' = Down, 'L' = Left, 'R' = Right.


    Raises
    ------
    IndexError:
        If the starting or the ending patch are not in the image.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates.

    """
    delta_x = np.zeros(shape)
    delta_y = np.zeros(shape)

    if shift_direction == "R":
        delta_x[ul[0] : ul[0] + height, ul[1] : ul[1] + width] = shift_size

    elif shift_direction == "L":
        delta_x[ul[0] : ul[0] + height, ul[1] : ul[1] + width] = -shift_size

    elif shift_direction == "U":
        delta_y[ul[0] : ul[0] + height, ul[1] : ul[1] + width] = -shift_size

    elif shift_direction == "D":
        delta_y[ul[0] : ul[0] + height, ul[1] : ul[1] + width] = shift_size

    else:
        raise ValueError("Unrecognized shift direction {}".format(shift_direction))

    return delta_x, delta_y


def projective(shape, matrix=None):
    """Projective transformation encoded in a 3 x 3 matrix.

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).

    matrix : np.ndarray
        Transformation matrix of the shape 3 x 3.

    Raises
    ------
    ValueError
        In case the transformation matrix has a wrong shape.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates.

    """
    if matrix is None:
        matrix = np.eye(3)

    if matrix.shape != (3, 3):
        raise ValueError(
            "The shape of projective transformation matrix is {}, correct is (3, 3)".format(
                matrix.shape
            )
        )

    tform = ProjectiveTransform(matrix)

    x, y = np.meshgrid(range(shape[1]), range(shape[0]))
    coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))).astype(float)

    coords_after = tform(coords)
    coords_delta = coords_after - coords

    delta_x = np.reshape(coords_delta[:, 0], shape)
    delta_y = np.reshape(coords_delta[:, 1], shape)

    return delta_x, delta_y


def single_frequency(
    shape,
    p=None,
    grid_spacing=5,
    n_perturbation_points=20,
    radius_mean=None,
    interpolation_method="griddata",
    interpolator_kwargs=None,
):
    """Single frequency artificial warping generator.

    Notes
    -----
    1) The reason why this approach is called single frequency is that the grid_spacing is constant over the entire
        image regions. One can therefore capture displacements of more or less the same size pixelwise.

    2) Calls the `control_points` so in this sense it is slightly unusual.

    3) The idea is to create a regular grid and then sample some of the points on it. For a fixed point
        we then randomly select and angle - Uniform[0, 2pi] and also the diameter ~ Exp(`radius_mean`)

    Parameters
    ----------
    shape : tuple
        Of the form (height, width).

    p : np.array, optional
        Pixelwise probability of selection, where p.shape=shape. Note that if None, then uniform.

    grid_spacing : int, optional
        Grid spacing size in both the columns and rows.

    n_perturbation_points : int, optional
        Number of grid points to which random perturbation will be applied (without replacement).

    radius_mean : float, optional
        If None then set to `grid_spacing / 2`.

    interpolation_method : {'griddata', 'bspline', 'rbf'}, optional
        Interpolation method to use.

    interpolator_kwargs : dict, optional
        Additional parameters passed to the interpolator.

    Returns
    -------
    delta_x : np.ndarray
        Displacement vector field of the x coordinates.

    delta_y : np.ndarray
        Displacement vector field of the y coordinates.

    References
    ----------
    [1] https://github.com/hsokooti/RegNet (README.md)

    """
    height, width = shape
    pixel_count = np.prod(shape)
    radius_mean = grid_spacing / 2 if radius_mean is None else radius_mean

    # Create grid mask (True = on grid, False = off grid)
    ixs_x, ixs_y = (
        list(range(width))[::grid_spacing],
        list(range(height))[::grid_spacing],
    )

    grid_mask = np.zeros(shape, dtype=bool)
    grid_mask[ixs_y, :] = True
    grid_mask[:, ixs_x] = True

    grid_pixel_count = np.sum(grid_mask)

    if grid_pixel_count < n_perturbation_points:
        raise ValueError("Cannot have more perturbation points than grid points.")

    if p is not None:
        if not p.sum() == 1:
            raise ValueError("The p needs to sum up to one or be None.")

        p_c = p.copy()
        sum_offgrid = np.sum(p_c[~grid_mask])
        p_c[~grid_mask] = 0  # make it impossible to choose off grid elements
        p_c[grid_mask] += sum_offgrid / np.sum(grid_mask)

    else:
        p_c = np.zeros(shape)
        p_c[grid_mask] = 1 / np.sum(grid_mask)

    p_ = p_c.ravel()

    selected_pixels = np.random.choice(
        pixel_count, n_perturbation_points, replace=False, p=p_
    )
    selected_pixels_rc = np.array([(x // width, x % width) for x in selected_pixels])
    perturbation_radii = np.random.exponential(radius_mean, n_perturbation_points)
    perturbation_angles = np.random.uniform(
        0, 2 * np.pi, n_perturbation_points
    )  # in radians

    pixel_displacements_rc = np.array(
        [
            np.array([-np.sin(angle), np.cos(angle)]) * radius
            for radius, angle in zip(perturbation_radii, perturbation_angles)
        ]
    )

    return control_points(
        shape=shape,
        points=selected_pixels_rc,
        values_delta_x=pixel_displacements_rc[:, 0],
        values_delta_y=pixel_displacements_rc[:, 1],
        interpolation_method=interpolation_method,
        interpolator_kwargs=interpolator_kwargs,
    )
