"""A module implementing some useful metrics.

Notes
-----
The metrics depend on what we apply them to. In general there are 4 types of metrics:
    * DVF (=multioutput regression)
    * Annotation (=per pixel classification)
    * Image
    * Keypoint metrics

These metrics are using numpy and are supposed to be used locally after forward passing

In DVF Metrics, the word 'combined' denotes the fact that we are performing multioutput regression. Each of the metrics
should always return a tuple of (metric_average, metric_per_output).

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

from functools import wraps

import ants

try:
    import lpips_tf
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        """
        LPIPS-TensorFlow required but not found.

        Please install it by running the following command:
        $ pip install git+http://github.com/alexlee-gk/lpips-tensorflow.git#egg=lpips_tf
        """
    ) from err

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial import distance
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from atlalign.base import DisplacementField
from atlalign.data import annotation_volume, segmentation_collapsing_labels
from atlalign.utils import find_labels_dic


# General utils
def _checker_and_convertor_dvf(y_true, y_pred, copy=True):
    """Check if inputs correct and convert to np.ndarray if necessary.

    Parameters
    ----------
    y_true : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    copy : bool
        If True, then copied arrays are returned.

    Returns
    -------
    y_true : np.ndarray
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.

    """
    inputs = {"y_true": y_true, "y_pred": y_pred}
    # checks
    for y_name, y in inputs.items():
        if not isinstance(y, (list, np.ndarray)):
            raise TypeError(
                "The {} needs to be either a list or np.ndarray, current type:{}".format(
                    y_name, type(y)
                )
            )

        if isinstance(y, list):
            if not np.all([isinstance(df, DisplacementField) for df in y]):
                raise TypeError(
                    "All elements of the list of {} need to be an instance of DisplacementField"
                )

            # convert to np.ndarray
            y_array = np.array([np.stack((df.delta_x, df.delta_y), axis=2) for df in y])

            return (
                _checker_and_convertor_dvf(y_true, y_array, copy=copy)
                if y_name == "y_pred"
                else _checker_and_convertor_dvf(y_array, y_pred, copy=copy)
            )

        if y.ndim != 4:
            raise ValueError(
                "The number of dimensions of {} needs to be 4.".format(y_name)
            )

        if y.shape[3] != 2:
            raise ValueError("The displacement field needs to have delta_x and delta_y")

    if y_true.shape != y_pred.shape:
        raise ValueError("The two input arrays have inconsistent shapes")

    if copy:
        return y_true.copy(), y_pred.copy()

    else:
        return y_true, y_pred


def _scikit_metric_wrapper(y_true, y_pred, metric_callable, **kwargs):
    """Wrap scikit regression metrics so that it is possible to work with image shaped regression outputs.

    Parameters
    ----------
    y_true : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    metric_callable : callable
        A scikit-learn-like metric that inputs y_true, y_pred of shapes (n_samples, n_outputs) and outputs
        (n_outputs, ) metric scores.

    kwargs : dict
        Keyword arguments passed into a corresponding scikit-learns metric.

    Returns
    -------
    metric_average : float
        An average version of the metric over all regression outputs.

    metric_per_output : np.ndarray
        An np.ndarray of shape (h, w, 2) representing individual metric scores for each regression output.

    """
    y_true, y_pred = _checker_and_convertor_dvf(y_true, y_pred)

    kwargs.update({"multioutput": "raw_values"})

    shape = y_true.shape
    N = shape[0]

    res_raw = metric_callable(
        y_true.reshape((N, -1)), y_pred.reshape((N, -1)), **kwargs
    )

    metric_per_output = res_raw.reshape(shape[1:])
    metric_average = metric_per_output.mean()

    return metric_average, metric_per_output


# OPTICAL FLOW METRICS
def angular_error_of(y_true, y_pred, weighted=False):
    """Compute angular error between two displacement fields.

    Parameters
    ----------
    y_true : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    weighted : bool
        Only applicable in cases where `N=1`. The norm of `y_true` is used to created the weights for the average.

    Returns
    -------
    angular_error_average : float
        An average angular error over all samples and pixels. If `weighted=True` then it is a weighted average
        where the weights are derived from the norm of the `y_true`.

    angular_error_per_output : np.ndarray
        An np.ndarray of shape (h, w) representing an average over all samples of angular error.
    """
    y_true, y_pred = _checker_and_convertor_dvf(y_true, y_pred)

    shape = y_true.shape

    if weighted and shape[0] != 1:
        raise ValueError("The weighted average is only allowed for a single sample.")

    angular_error_per_output_per_sample = np.zeros(shape[:-1])

    for i in range(shape[0]):
        delta_x_true = y_true[i, ..., 0]
        delta_y_true = y_true[i, ..., 1]

        delta_x_pred = y_pred[i, ..., 0]
        delta_y_pred = y_pred[i, ..., 1]

        top = delta_x_true * delta_x_pred + delta_y_true * delta_y_pred
        bottom = np.sqrt(
            delta_x_true * delta_x_true + delta_y_true * delta_y_true
        ) * np.sqrt(delta_x_pred * delta_x_pred + delta_y_pred * delta_y_pred)

        angular_error_per_output_per_sample[i] = np.rad2deg(
            np.arccos(np.clip(top / bottom, -1, 1))
        )

    angular_error_per_output = np.nanmean(
        angular_error_per_output_per_sample, axis=0
    )  # (h, w)

    if weighted:
        norm = DisplacementField(y_true[0, ..., 0], y_true[0, ..., 1]).norm
        weights = norm / norm.sum()
        angular_error_average = np.where(
            np.isnan(angular_error_per_output), 0, angular_error_per_output * weights
        ).sum()  # ehm, true and pred different nan
    else:
        angular_error_average = np.nanmean(angular_error_per_output)  # float

    return angular_error_average, angular_error_per_output


# REGRESSION METRICS
def correlation_combined(y_true, y_pred):
    """Compute combined version of correlation.

    Notes
    -----
    Slow.

    Parameters
    ----------
    y_true : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    Returns
    -------
    correlation_average : float
        Mean correlation.

    correlation_per_output : np.ndarray
        An np.ndarray of shape (h, w, 2) representing individual correlation scores for each regression output.

    """
    y_true, y_pred = _checker_and_convertor_dvf(y_true, y_pred)

    shape = y_true.shape

    df_true = pd.DataFrame(y_true.reshape(len(y_true), -1))
    df_pred = pd.DataFrame(y_pred.reshape(len(y_pred), -1))

    res_s = df_true.apply(lambda x: np.corrcoef(x, df_pred[x.name])[0, 1], axis=0)
    # res_s = df_true.apply(lambda x: x.corr(df_pred[x.name]), axis=0)

    correlation_per_output = res_s.values.reshape(shape[1:])

    correlation_average = correlation_per_output.mean()

    return correlation_average, correlation_per_output


def mae_combined(y_true, y_pred):
    """Compute combined version of mean absolute error.

    Notes
    -----
    A difference between this implementation and scikit-learn is that the inputs here `y_true`, `y_pred` are
    custom made for our registration problem.

    Parameters
    ----------
    y_true : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    Returns
    -------
    mae_average : float
        A combined mse.

    mae_per_output : np.ndarray
        An np.ndarray of shape (h, w, 2) representing individual mae scores for each regression output.

    """
    return _scikit_metric_wrapper(y_true, y_pred, metric_callable=mean_absolute_error)


def mse_combined(y_true, y_pred):
    """Compute combined version of mean squared error.

    Notes
    -----
    A difference between this implementation and scikit-learn is that the inputs here `y_true`, `y_pred` are
    custom made for our registration problem.

    Parameters
    ----------
    y_true : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    Returns
    -------
    mse_average : float
        A combined mse.

    mse_per_output : np.ndarray
        An np.ndarray of shape (h, w, 2) representing individual mse scores for each regression output.

    """
    return _scikit_metric_wrapper(y_true, y_pred, metric_callable=mean_squared_error)


def r2_combined(y_true, y_pred):
    """Compute combined version of r2.

    Notes
    -----
    A difference between this implementation and scikit-learn is that the inputs here `y_true`, `y_pred` are
    custom made for our registration problem.

    Parameters
    ----------
    y_true : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    Returns
    -------
    r2_average : float
        A combined r2.

    r2_per_output : np.ndarray
        An np.ndarray of shape (h, w, 2) representing individual r2 scores for each regression output.

    """
    return _scikit_metric_wrapper(y_true, y_pred, metric_callable=r2_score)


def vector_distance_combined(y_true, y_pred):
    """Compute combined version of vector distance.

    Parameters
    ----------
    y_true : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    y_pred : np.ndarray or list
        If np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.
        If list then elements are instances of ``atlalign.base.DisplacementField``.

    Returns
    -------
    vector_distance_average : float
        An average vector distance over all samples and pixels.

    vector_distance_per_output : np.ndarray
        An np.ndarray of shape (h, w) representing an average over all samples of vector distance.

    """
    y_true, y_pred = _checker_and_convertor_dvf(y_true, y_pred)

    diff = y_pred - y_true

    vector_distance_per_output = np.sqrt(
        np.square(diff[..., 0]) + np.square(diff[..., 1])
    ).mean(axis=0)
    vector_distance_average = vector_distance_per_output.mean()

    return vector_distance_average, vector_distance_per_output


# ANNOTATION METRICS


def _checker_annotation(y_true, y_pred, k):
    """Check whether the inputs for annotation are correct.

    Parameters
    ----------
    y_true : np.ndarray
        A np.ndarray of shape (N, h, w) such that the first dimension represents the sample.
        The ground truth annotation.

    y_pred : np.ndarray
        A np.ndarray of shape (N, h, w) such that the first dimensions represents the sample.
        The predicted annotation.

    k : int or float or None
        A class label. If None, then averaging based on label distribution in each true image is
        performed.

    Raises
    ------
    TypeError
        If `y_true` or `y_pred` not ``np.ndarray``.

    ValueError
        Various inconsistencies.

    """
    if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
        raise TypeError("Both of the inputs y_true and y_pred need to be a np.ndarray.")

    if not (y_true.ndim == 3 and y_pred.ndim == 3):
        raise ValueError("The input arrays need to be 3D.")

    if y_true.shape != y_pred.shape:
        raise ValueError("The arrays have different shapes.")

    # Make sure k appears at least once in at least one of the arrays or its a None
    if k is None:
        return

    if not (np.any(y_true == k) or np.any(y_pred == k)):
        raise ValueError("The k={} is not present in either of the arrays.".format(k))


def iou_score(y_true, y_pred, k=0, disable_check=False, excluded_labels=None):
    """Compute intersection over union of a class `k` equally weighted over all samples.

    Notes
    -----
    If the class is not present in either of the images — ('0/0') type of situation, we simply skip
    this sample and do not consider it if for the average.

    Parameters
    ----------
    y_true : np.ndarray
        A np.ndarray of shape (N, h, w) such that the first dimension represents the sample.
        The ground truth annotation.

    y_pred : np.ndarray
        A np.ndarray of shape (N, h, w) such that the first dimensions represents the sample.
        The predicted annotation.

    k : int or float or None
        A class label. If None, then averaging based on label distribution in each true image is
        performed.

    disable_check : bool
        If False, then checks are disabled. Used when recursively calling the function.

    excluded_labels : None or list
        If None then no effect. If a list of ints then they wont be used in the averaging over labels (in case
        `k` is None).

    Returns
    -------
    iou_average : float
        An average IOU over all samples.

    iou_per_sample : np.ndarray, shape = (N,)
        IOU score per each sample. Note that if label does not occur in either of the images then
        its equal to np.nan.

    """
    # Run checks
    if not disable_check:
        _checker_annotation(y_true, y_pred, k)

    n = len(y_true)
    # n_pixels = np.prod(y_true.shape[1:])

    iou_per_sample = np.zeros(n, dtype=np.float32)

    for i in range(n):
        n_pixels = (
            ~np.isin(y_true[i], np.array(excluded_labels or []))
        ).sum()  # only non excluded pixels

        if k is not None:
            mask_true = y_true[i] == k
            mask_pred = y_pred[i] == k

            intersection = np.logical_and(mask_true, mask_pred)
            union = np.logical_or(mask_true, mask_pred)

            iou_per_sample[i] = (
                (intersection.sum() / union.sum()) if not np.all(union == 0) else np.nan
            )

        else:
            # true image distribution
            w = {
                label: (y_true[i] == label).sum() / n_pixels
                for label in (set(np.unique(y_true[i])) | set(np.unique(y_pred[i])))
                - set(excluded_labels or [])
            }

            weighted_average = sum(
                iou_score(
                    y_true[i][np.newaxis, :, :],
                    y_pred[i][np.newaxis, :, :],
                    k,
                    disable_check=True,
                )[0]
                * p
                for k, p in w.items()
            )

            iou_per_sample[i] = weighted_average

    iou_average = np.nanmean(iou_per_sample)

    return iou_average, iou_per_sample


def dice_score(y_true, y_pred, k=0, disable_check=False, excluded_labels=None):
    """Compute dice score of a class `k` equally weighted over all samples.

    Notes
    -----
    If the class is not present in either of the images — ('0/0') type of situation, we simply skip
    this sample and do not consider it if for the average.

    Parameters
    ----------
    y_true : np.ndarray
        A np.ndarray of shape (N, h, w) such that the first dimension represents the sample.
        The ground truth annotation.

    y_pred : np.ndarray
        A np.ndarray of shape (N, h, w) such that the first dimensions represents the sample.
        The predicted annotation.

    k : int or float or None
        A class label. If None, then averaging based on label distribution in each true image is
        performed.

    disable_check : bool
        If False, then checks are disabled. Used when recursively calling the function.

    excluded_labels : None or list
        If None then no effect. If a list of ints then they wont be used in the averaging over labels (in case
        `k` is None).

    Returns
    -------
    dice_average : float
        An average dice over all samples.

    dice_per_sample : np.ndarray, shape = (N,)
        Dice score per each sample. Note that if label does not occur in either of the images then
        its equal to np.nan.

    """
    # Run checks
    if not disable_check:
        _checker_annotation(y_true, y_pred, k)

    n = len(y_true)
    # n_pixels = np.prod(y_true.shape[1:])

    dice_per_sample = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if set(np.unique(y_true[i])) != set(np.unique(y_pred[i])):
            # print('Different labels')
            pass

        n_pixels = (
            ~np.isin(y_true[i], np.array(excluded_labels or []))
        ).sum()  # only non excluded pixels

        if k is not None:
            mask_true = y_true[i] == k
            mask_pred = y_pred[i] == k

            intersection = np.logical_and(mask_true, mask_pred)
            union = np.logical_or(mask_true, mask_pred)

            iou_per_sample = (
                intersection.sum() / union.sum() if not np.all(union == 0) else np.nan
            )
            dice_per_sample[i] = (2 * iou_per_sample) / (1 + iou_per_sample)
        else:
            # true image distribution
            w = {
                label: (y_true[i] == label).sum() / n_pixels
                for label in (set(np.unique(y_true[i])) | set(np.unique(y_pred[i])))
                - set(excluded_labels or [])
            }

            weighted_average = sum(
                dice_score(
                    y_true[i][np.newaxis, :, :],
                    y_pred[i][np.newaxis, :, :],
                    k,
                    disable_check=True,
                )[0]
                * p
                for k, p in w.items()
            )

            dice_per_sample[i] = weighted_average

    dice_average = np.nanmean(dice_per_sample)

    return dice_average, dice_per_sample


# IMAGE METRICS
def multiple_images_decorator(fun):
    """Enhance a function with iteration over the samples.

    Parameters
    ----------
    fun : callable
        Callable whose functionality will be enhanced.


    Returns
    -------
    wrapper_fun : callable
        Enhanced version of `fun` callable.

    """
    # make sure metadata kept
    @wraps(fun)
    def wrapper_fun(*args, **kwargs):
        """Define enhanced function."""
        assert len(args) == 2

        y_true, y_pred = args

        # CHECKS
        if not (y_pred.shape[-2:] == y_true.shape[-2:]):
            raise ValueError("The image shape is not consistent.")

        if not (y_pred.dtype == np.float32 and y_true.dtype == np.float32):
            raise TypeError("The only allowed dtype is float32.")

        if kwargs.get("mask") is not None:
            if kwargs["mask"].shape != y_true.shape[-2:]:
                raise ValueError(
                    "The mask has to have the same shape as the input images."
                )

            if kwargs["mask"].dtype != np.bool:
                raise ValueError("The mask needs to be an array of booleans.")

        # MAIN ALGORITHM
        if y_true.ndim == 2:
            return fun(*args, **kwargs)

        elif y_true.ndim == 3:

            if y_true.shape[0] != y_pred.shape[0]:
                raise ValueError(
                    "The number of y_true images and y_pred images has to be the same"
                )

            metric_values = []

            for i in range(y_true.shape[0]):
                y_true_single = y_true[i]
                y_pred_single = y_pred[i]

                metric_values.append(fun(y_true_single, y_pred_single, **kwargs))

            return np.array(metric_values)

        else:
            raise ValueError("Invalid number of dimensions {}".format(y_true.ndim))

    return wrapper_fun


@multiple_images_decorator
def mse_img(y_true, y_pred, mask=None):
    """Compute the mean-squared error between two images.

    Parameters
    ----------
    y_true : np.array
        Image 1. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    y_pred : np.array
        Image 2. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    mask: np.array, optional
        Optional, can be specified to have the computation carried out on a precise area.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric. Loss metric, the lower the more similar the images are.

    """
    if mask is None:
        size = y_true.shape
        mask = np.ones(size, dtype=bool)

    return np.sum((y_true[mask] - y_pred[mask]) ** 2) / np.sum(mask)


@multiple_images_decorator
def mae_img(y_true, y_pred, mask=None):
    """Compute the mean absolute error between two images.

    Parameters
    ----------
    y_true : np.array
        Image 1. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    y_pred : np.array
        Image 2. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    mask: np.array, optional
        Optional, can be specified to have the computation carried out on a precise area.

    Returns
    -------
    mse : float
        The mean absolute error (MAE) metric. Loss metric, the lower the more similar the images are.

    """
    if mask is None:
        size = y_true.shape
        mask = np.ones(size, dtype=bool)

    return np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(mask)


@multiple_images_decorator
def psnr_img(y_true, y_pred, mask=None, data_range=None):
    """Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    y_true : np.array
        Image 1. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    y_pred : np.array
        Image 2. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    mask: np.array, optional
        Optional, can be specified to have the computation carried out on a precise area.

    data_range : int
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric. Similarity metric, the higher the more similar the images are.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    if mask is None:
        size = y_true.shape
        mask = np.ones(size, dtype=bool)

    return peak_signal_noise_ratio(y_true[mask], y_pred[mask], data_range=data_range)


@multiple_images_decorator
def demons_img(y_true, y_pred, mask=None):
    """Compute the demons metric between two images.

    Parameters
    ----------
    y_true : np.array
        Image 1. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    y_pred : np.array
        Image 2. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    mask: np.array, optional
        Optional, can be specified to have the computation carried out on a precise area.

    Returns
    -------
    demons : float
        The demons value.  Loss metric, the lower the more similar the images are.

    """
    y_true_ants = ants.image_clone(ants.from_numpy(y_true), pixeltype="float")
    y_pred_ants = ants.image_clone(ants.from_numpy(y_pred), pixeltype="float")

    if mask is None:
        demons = ants.image_similarity(y_true_ants, y_pred_ants, metric_type="Demons")

    else:
        mask_ants = ants.image_clone(
            ants.from_numpy(mask.astype(float)), pixeltype="float"
        )
        demons = ants.image_similarity(
            y_true_ants,
            y_pred_ants,
            fixed_mask=mask_ants,
            moving_mask=mask_ants,
            metric_type="Demons",
        )

    return demons


@multiple_images_decorator
def cross_correlation_img(y_true, y_pred, mask=None):
    """Compute the cross correlation metric between two images.

    Parameters
    ----------
    y_true : np.array
        Image 1. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    y_pred : np.array
        Image 2. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    mask: np.array, optional
        Optional, can be specified to have the computation carried out on a precise area.

    Returns
    -------
    cc : float
        The Cross-Correlation value. Similarity metric, the higher the more similar the images are.

    """
    y_true_ants = ants.image_clone(ants.from_numpy(y_true), pixeltype="float")
    y_pred_ants = ants.image_clone(ants.from_numpy(y_pred), pixeltype="float")

    if mask is None:
        cc = ants.image_similarity(y_true_ants, y_pred_ants, metric_type="Correlation")

    else:
        mask_ants = ants.image_clone(
            ants.from_numpy(mask.astype(float)), pixeltype="float"
        )
        cc = ants.image_similarity(
            y_true_ants,
            y_pred_ants,
            fixed_mask=mask_ants,
            moving_mask=mask_ants,
            metric_type="Correlation",
        )

    return -cc


@multiple_images_decorator
def ssmi_img(y_true, y_pred):
    """Compute the structural similarity between two images.

    Parameters
    ----------
    y_true : np.array
        Image 1. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    y_pred : np.array
        Image 2. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    Returns
    -------
    ssmi : float
        The structural similarity (SSMI) metric. Similarity metric, the higher the more similar the images are.

    """
    return structural_similarity(y_true, y_pred)


@multiple_images_decorator
def mi_img(y_true, y_pred, mask=None, metric_type="MattesMutualInformation"):
    """Compute the mutual information (MI) between two images.

    Parameters
    ----------
    y_true : np.array
        Image 1. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    y_pred : np.array
        Image 2. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    mask: np.array, optional
        Optional, can be specified to have the computation carried out on a precise area.

    metric_type: str, {'MattesMutualInformation', 'JointHistogramMutualInformation'}
        Type of mutual information computation.

    Returns
    -------
    mi : float
        The mutual information (MI) metric. Similarity metric, the higher the more similar the images are.

    """
    y_true_ants = ants.image_clone(ants.from_numpy(y_true), pixeltype="float")
    y_pred_ants = ants.image_clone(ants.from_numpy(y_pred), pixeltype="float")

    if mask is None:
        mi = ants.image_similarity(y_true_ants, y_pred_ants, metric_type=metric_type)

    else:
        mask_ants = ants.image_clone(
            ants.from_numpy(mask.astype(float)), pixeltype="float"
        )
        mi = ants.image_similarity(
            y_true_ants,
            y_pred_ants,
            fixed_mask=mask_ants,
            moving_mask=mask_ants,
            metric_type=metric_type,
        )

    return -mi


@multiple_images_decorator
def perceptual_loss_img(y_true, y_pred, model="net-lin", net="vgg"):
    """Compute the perceptual loss (PL) between two images.

    Parameters
    ----------
    y_true : np.array
        Image 1. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    y_pred : np.array
        Image 2. Either (h, w) of (N, h, w). If (N, h, w), the decorator `multiple_images_decorator` takes care of
        the sample dimension.

    model: str, {'net', 'net-lin'}
        Type of model (cf lpips_tf package).

    net: str, {'vgg', 'alex'}
        Type of network (cf lpips_tf package).

    Return
    ------
    pl: float
        The Perceptual Loss (PL) metric. Loss metric, the lower the more similar the images are.

    Notes
    -----
    We use the decorator just to make sure we do not run out of memory during a forward pass. Also,
    its fully convolutional but if the images are too small then might run into issues.
    """
    # gray2rgb
    y_true = np.stack((y_true,) * 3, axis=-1)
    y_pred = np.stack((y_pred,) * 3, axis=-1)

    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)

    distance_t = lpips_tf.lpips(image0_ph, image1_ph, model=model, net=net)

    with tf.Session() as session:
        pl = session.run(distance_t, feed_dict={image0_ph: y_true, image1_ph: y_pred})

    tf.reset_default_graph()

    return pl.item()


# KEYPOINTS METRICS
def _euclidean_distance_kp(y_true, y_pred):
    """Euclidean distance between pairs of n-dimensional points.

    Parameters
    ----------
    y_true : np.array
        Array of shape (N, d) where `N` is the number of samples.

    y_pred : np.array
        Array of shape (N, d) where `N` is the number of samples.

    Returns
    -------
    distance : np.array
        Array of shape (N, ) representing the Euclidean distance between `y_true` and `y_pred` for each sample.

    Raises
    ------
    ValueError:
        In case the shapes do not agree.

    Examples
    --------
    >>> import numpy as np
    >>> from atlalign.metrics import _euclidean_distance_kp
    >>> np.random.seed(0)
    >>> y_true = np.array([[1, 2], [3.4, 4], [2, 1]])
    >>> y_pred = np.array([[1.2, 3], [3, 4], [2, 1.8]])
    >>> _euclidean_distance_kp(y_true, y_pred)
    array([1.0198039, 0.4      , 0.8      ])
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "The shapes of true and predicted are different, {} vs {}".format(
                y_true.shape, y_pred.shape
            )
        )

    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))


def _compute_weights_kp(y, distance_kwargs=None):
    """Compute weights of keypoints.

    Parameters
    ----------
    y : np.array
        Array of shape (N, 2) where each row represents a keypoint.

    distance_kwargs : dict or None
        If ``dict`` then represents parameters to be passed into scipy implementation.

    Returns
    -------
    weights : np.array
        Array of shape (N, ) where each element represents a weight of that specific keypoint.

    """
    dist_matrix = distance.cdist(
        y, y, **(distance_kwargs or {})
    )  # uses euclidean distance by default

    mean_dist = np.mean(dist_matrix, axis=0)

    return mean_dist / mean_dist.sum()


def tre_kp(y_true, y_pred, weighted=False):
    """Compute (absolute) target registration error.

    Parameters
    ----------
    y_true : np.array
        Array of shape (N, 2) where `N` is the number of samples.

    y_pred : np.array
        Array of shape (N, 2) where `N` is the number of samples.

    weighted : bool
        If True, then the final TRE is weighted by `y_true`.

    Returns
    -------
    mean_tre : float
        Mean target registration error over all keypoint pairs.

    tre : np.array
        Array of shape (N,) where elements represent the TRE of a given keypoint par.

    weights : np.array
        Array of shape (N,). If `weighted` is False then simply 1 / N. Otherwise the weights are derived from `y_true`.

    References
    ----------
    .. [1] https://anhir.grand-challenge.org/Performance_Metrics/

    """
    # Extract useful parameters
    n = len(y_true)

    tre = _euclidean_distance_kp(y_true, y_pred)
    weights = _compute_weights_kp(y_true) if weighted else 1 / n * np.ones(n)
    mean_tre = np.inner(tre, weights)

    return mean_tre, tre, weights


def rtre_kp(y_true, y_pred, h, w, weighted=False):
    """Compute relative target registration error.

    Parameters
    ----------
    y_true : np.array
        Array of shape (N, 2) where `N` is the number of samples.

    y_pred : np.array
        Array of shape (N, 2) where `N` is the number of samples.

    h : np.array
        Height of the image.

    w : np.array
        Width of the image.

    weighted : bool
        If True, then the final TRE is weighted by `y_true`.

    Returns
    -------
    mean_tre : float
        Mean target registration error over all keypoint pairs.

    rtre : np.array
        Array of shape (N,) where elements represent the rTRE of a given keypoint par.

    weights : np.array
        Array of shape (N,). If `weighted` is False then simply 1 / N. Otherwise the weights are derived from `y_true`.

    References
    ----------
    .. [1] https://anhir.grand-challenge.org/Performance_Metrics/

    """
    # Extract useful parameters
    n = len(y_true)

    diagonal = (w ** 2 + h ** 2) ** (1 / 2)

    rtre = _euclidean_distance_kp(y_true, y_pred) / diagonal
    weights = _compute_weights_kp(y_true) if weighted else 1 / n * np.ones(n)
    mean_rtre = np.inner(rtre, weights)

    return mean_rtre, rtre, weights


def improvement_kp(y_true, y_pred, y_init):
    """Compute improvement ratio with respect to initial keypoints.

    Parameters
    ----------
    y_true : np.array
        Array of shape (N, 2) where `N` is the number of samples. Ground truth positions in the reference space.

    y_pred : np.array
        Array of shape (N, 2) where `N` is the number of samples. Predicted positions in the reference space.

    y_init : np.array
        Array of shape (N, 2) where `N` is the number of samples. Initial positions in the reference space.

    Returns
    -------
    percent_improved : float
        Percent of predicted keypoints that were better than the initial ones.

    mask : np.array
        Array of booleans representing which predicted keypoints achieved a higher TRE than the inital one.

    """
    pred_vs_true = tre_kp(y_true, y_pred)[1]
    init_vs_true = tre_kp(y_true, y_init)[1]

    mask = pred_vs_true < init_vs_true
    percent_improved = mask.sum() / len(mask)

    return percent_improved, mask


# ALL TOGETHER
def evaluate(y_true, y_pred, imgs_mov, img_ids, ps, dataset_ids, depths=()):
    """Evaluate all relevant matrics in per sample fashion.

    Parameters
    ----------
    y_true : np.ndarray
        np.ndarray then expected shape is (N, h, w, 2) and represents true samples of displacement fields.

    y_pred : np.ndarray
        np.ndarray then expected shape is (N, h, w, 2) and represents predicted samples of displacement fields.

    imgs_mov : np.ndarray
        Array of shape (N, h, w) representing the moving images. If dtype=float32 then no division. If uint8
        then values devided by 255 and casted to flaot32.

    img_ids : np.array
        Array of shape (N, ) representing the image ids.

    dataset_ids : np.array
        Array of shape (N, ) representing the dataset ids.

    depths : tuple
        Tuple of different depths to compute the intersection over union score.

    Returns
    -------
    result : pd.DataFrame
        All results even containing array entries.

    results_viewable : pd.DataFrame
        Results without array entries.
    """
    if (
        len(y_true)
        != len(y_pred)
        != len(imgs_mov)
        != len(img_ids)
        != len(ps)
        != len(dataset_ids)
    ):
        raise ValueError()
    n = len(y_true)

    imgs_mov = [
        imgs_mov[i].astype(np.float32) / (255 if imgs_mov.dtype == np.uint8 else 1)
        for i in range(n)
    ]

    # displacement fields
    df_true = [
        DisplacementField(y_true[i, ..., 0], y_true[i, ..., 1]) for i in range(n)
    ]
    df_pred = [
        DisplacementField(y_pred[i, ..., 0], y_pred[i, ..., 1]) for i in range(n)
    ]

    # inverse dvfs
    inv_df_true = [df.pseudo_inverse(ds_f=64) for df in df_true]
    inv_df_pred = []
    for df in df_pred:
        try:
            inv_df_pred.append(df.pseudo_inverse(ds_f=64))
        except Exception:
            inv_df_pred.append(None)

    # image registration
    imgs_reg_true = [df_true[i].warp(imgs_mov[i]) for i in range(n)]
    imgs_reg_pred = [df_pred[i].warp(imgs_mov[i]) for i in range(n)]

    # compute norms pred
    template_dict = {
        # SINGLE DVFS
        "norm_a": pd.Series(0, index=range(n), dtype=float),
        "norm_pp": [np.empty((320, 456)) for _ in range(n)],
        "jacobian_nonpositive_pixels_a": pd.Series(0, index=range(n), dtype=float),
        "jacobian_pp": [np.empty((320, 456)) for _ in range(n)],
        # META
        "p": pd.Series(ps, dtype=int),
        "dataset_id": pd.Series(dataset_ids, dtype=int),
        "imgs_mov": imgs_mov,
        # OPTICAL FLOW
        "vector_distance_a": pd.Series(0, index=range(n), dtype=float),
        "vector_distance_pp": [np.empty((320, 456)) for _ in range(n)],
        "angular_error_a": pd.Series(0, index=range(n), dtype=float),
        "angular_error_pp": [np.empty((320, 456)) for _ in range(n)],
        # REGRESSION METRICS
        #                     'correlation_a': pd.Series(0, index=range(n), dtype=float),
        #                     'correlation_po':  [np.empty((320, 456, 2)) for _ in range(n)],
        # IMAGE METRICS
        "mae_img_a": pd.Series(0, index=range(n), dtype=float),
        "mse_img_a": pd.Series(0, index=range(n), dtype=float),
        "psnr_img_a": pd.Series(0, index=range(n), dtype=float),
        "cross_correlation_img_a": pd.Series(0, index=range(n), dtype=float),
        "ssmi_img_a": pd.Series(0, index=range(n), dtype=float),
        "mi_img_a": pd.Series(0, index=range(n), dtype=float),
        # Segmentation metrics
        #                     'iou_depth_0' : None
    }

    # NORM
    for i in range(n):
        template_dict["norm_pp"][i] = df_pred[i].norm
        template_dict["norm_a"][i] = template_dict["norm_pp"][i].mean()

    # JACOBIAN
    for i in range(n):
        template_dict["jacobian_pp"][i] = df_pred[i].jacobian
        template_dict["jacobian_nonpositive_pixels_a"][i] = np.sum(
            template_dict["jacobian_pp"][i] <= 0
        )

    template_dict["jacobian_nonpositive_pixels_perc_a"] = (
        100 * template_dict["jacobian_nonpositive_pixels_a"] / (320 * 456)
    )  # noqa
    # VECTOR DISTANCE
    for i in range(n):
        vd_a, vd_pp = vector_distance_combined([df_true[i]], [df_pred[i]])
        (
            template_dict["vector_distance_a"][i],
            template_dict["vector_distance_pp"][i],
        ) = (vd_a, vd_pp)

    # VECTOR DISTANCE
    for i in range(n):
        ae_a, ae_pp = angular_error_of([df_true[i]], [df_pred[i]], weighted=True)
        template_dict["angular_error_a"][i], template_dict["angular_error_pp"][i] = (
            ae_a,
            ae_pp,
        )

    # MSE IMAGE
    template_dict["mae_img_a"] = mae_img(
        np.array(imgs_reg_true), np.array(imgs_reg_pred)
    )
    template_dict["mse_img_a"] = mse_img(
        np.array(imgs_reg_true), np.array(imgs_reg_pred)
    )
    template_dict["psnr_img_a"] = psnr_img(
        np.array(imgs_reg_true), np.array(imgs_reg_pred)
    )
    template_dict["ssmi_img_a"] = ssmi_img(
        np.array(imgs_reg_true), np.array(imgs_reg_pred)
    )
    template_dict["mi_img_a"] = mi_img(np.array(imgs_reg_true), np.array(imgs_reg_pred))
    template_dict["cross_correlation_img_a"] = cross_correlation_img(
        np.array(imgs_reg_true), np.array(imgs_reg_pred)
    )

    # Segmentation metrics
    if depths:
        annot_vol = annotation_volume()
        labels_dict = segmentation_collapsing_labels()
        for d in depths:
            print("Depth {}".format(d))
            ious = []
            for i, p in enumerate(ps):
                segm_ref = find_labels_dic(annot_vol[p // 25], labels_dict, d)

                segm_true = inv_df_true[i].warp_annotation(segm_ref)
                if inv_df_pred[i] is not None:
                    segm_pred = inv_df_pred[i].warp_annotation(segm_ref)
                    ious.append(
                        iou_score(
                            np.array([segm_true]),
                            np.array([segm_pred]),
                            k=None,
                            excluded_labels=[0],
                        )[0]
                    )
                else:
                    ious.append(np.nan)

            template_dict["iou_depth_{}".format(d)] = ious

    result = pd.DataFrame(template_dict)
    result = result.set_index(img_ids)

    return result, result[result.dtypes[result.dtypes != object].index]


def evaluate_single(
    deltas_true,
    deltas_pred,
    img_mov,
    p=None,
    avol=None,
    collapsing_labels=None,
    deltas_pred_inv=None,
    deltas_true_inv=None,
    ds_f=4,
    depths=(),
):
    """Evaluate a single sample.

    Parameters
    ----------
    deltas_true : DisplacementField or np.ndarray
        If np.ndarray then of shape (height, width, 2) representing deltas_xy of ground truth.

    deltas_pred : DisplacementField or np.ndarray
        If np.ndarray then of shape (height, width, 2) representing deltas_xy of prediction.

    img_mov : np.ndarray
        Moving image.

    p : int
        Coronal section in microns.

    avol : np.ndarray or None
        Annotation volume of shape (528, 320, 456). If None then loaded via `annotation_volume`.

    collapsing_labels : dict or None
        Dictionary for segmentation collapsing. If None then loaded via `segmentation_collapsing_labels`

    deltas_pred_inv : None or np.ndarray
        If np.ndarray then of shape (height, width, 2) representing inv_deltas_xy of prediction. If not provided
        computed from `df_pred`.

    deltas_true_inv : None or np.ndarray
        If np.ndarray then of shape (height, width, 2) representing inv_deltas_xy of truth. If not provided
        computed from `df_true`.

    ds_f : int
        Downsampling factor for numerical inversses.

    depths : tuple
        Tuple of integers representing all depths to compute IOU for. If empty no IOU computation takes places.

    Returns
    -------
    results : pd.Series
        Relevant metrics.
    """
    n_pixels = 320 * 456
    if not (deltas_true.shape == (320, 456, 2) and deltas_pred.shape == (320, 456, 2)):
        raise ValueError("Incorrect shape of input")

    df_true = DisplacementField(deltas_true[..., 0], deltas_true[..., 1])
    df_pred = DisplacementField(deltas_pred[..., 0], deltas_pred[..., 1])

    img_reg_true = df_true.warp(img_mov)
    img_reg_pred = df_pred.warp(img_mov)

    all_metrics = {
        "mse_img": mse_img(img_reg_true, img_reg_pred),
        "mae_img": mae_img(img_reg_true, img_reg_pred),
        "psnr_img": psnr_img(img_reg_true, img_reg_pred),
        "ssmi_img": ssmi_img(img_reg_true, img_reg_pred),
        "mi_img": mi_img(img_reg_true, img_reg_pred),
        "cc_img": cross_correlation_img(img_reg_true, img_reg_pred),
        # 'perceptual_img': perceptual_loss_img(img_reg_true, img_reg_pred),
        "norm": df_pred.norm.mean(),
        "corrupted_pixels": np.sum(df_pred.jacobian < 0) / n_pixels,
        "euclidean_distance": vector_distance_combined([df_true], [df_pred])[0],
        "angular_error": angular_error_of([df_true], [df_pred], weighted=True)[0],
    }

    # segmentations metrics
    if not depths:
        return all_metrics
    else:
        # checks
        if avol is None or avol.shape != (528, 320, 456):
            raise ValueError("Incorrectly shaped annotation volume or not provided")

        # Prepare inverses
        if deltas_true_inv is not None:
            df_true_inv = DisplacementField(
                deltas_true_inv[..., 0], deltas_true_inv[..., 1]
            )
        else:
            df_true_inv = df_true.pseudo_inverse(ds_f=ds_f)

        if deltas_pred_inv is not None:
            df_pred_inv = DisplacementField(
                deltas_pred_inv[..., 0], deltas_pred_inv[..., 1]
            )
        else:
            df_pred_inv = df_pred.pseudo_inverse(ds_f=ds_f)

        # Extract data
        avol_ = annotation_volume() if avol is None else avol
        collapsing_labels_ = (
            segmentation_collapsing_labels()
            if collapsing_labels is None
            else collapsing_labels
        )

        # Compute
        images = {}
        for depth in depths:
            segm_ref = find_labels_dic(avol_[p // 25], collapsing_labels_, depth)
            segm_true = df_true_inv.warp_annotation(segm_ref)
            segm_pred = df_pred_inv.warp_annotation(segm_ref)

            images[depth] = (segm_true, segm_pred)

            all_metrics["iou_{}".format(depth)] = iou_score(
                np.array([segm_true]),
                np.array([segm_pred]),
                k=None,
                excluded_labels=[0],
            )[0]

            all_metrics["dice_{}".format(depth)] = dice_score(
                np.array([segm_true]),
                np.array([segm_pred]),
                k=None,
                excluded_labels=[0],
            )[0]

        return all_metrics, images
