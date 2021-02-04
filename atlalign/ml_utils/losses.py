"""keras and Tensorflow implementation of different losses and metrics."""

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

import keras
import keras.backend as K

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
import tensorflow as tf


# IMAGE LOSSES
class DVF2IMG:
    """Class turning image losses into dvf losses."""

    def __init__(self, img_loss_callable, weights=(0.5, 0.5)):
        self.weights = weights
        self.img_loss_callable = img_loss_callable

    def loss(self, y_true, y_pred):
        """Compute loss."""
        delta_x_true = y_true[..., :1]
        delta_x_pred = y_pred[..., :1]
        delta_y_true = y_true[..., 1:]
        delta_y_pred = y_pred[..., 1:]

        delta_x_true, delta_x_pred = self.normalize(delta_x_true, delta_x_pred)
        delta_y_true, delta_y_pred = self.normalize(delta_y_true, delta_y_pred)

        loss_delta_x = self.img_loss_callable(delta_x_true, delta_x_pred)
        loss_delta_y = self.img_loss_callable(delta_y_true, delta_y_pred)

        return self.weights[0] * loss_delta_x + self.weights[1] * loss_delta_y

    @staticmethod
    def normalize(img_ref, img_other, c=10):
        """Divide by a constant."""
        return img_ref / c, img_other / c


class Mixer:
    """Mixes together multiple different losses."""

    def __init__(self, *args, weights=None):
        if not np.all([callable(x) for x in args]):
            raise TypeError("All the entries need to be callables.")

        n_args = len(args)

        if weights is None:
            weights = n_args * [1 / n_args]

        if len(weights) != n_args:
            raise ValueError("Weight needs to be provided for each callable.")

        self.callables = args
        self.weights = weights

    def loss(self, y_true, y_pred):
        """Compute loss."""
        result = 0
        for cal, w in zip(self.callables, self.weights):
            result += cal(y_true, y_pred) * w

        return result


class NCC:
    """Class computing normalized cross correlations under different hyperparameters."""

    def __init__(self, win=None, eps=1e-5):
        self.eps = eps

        if win is None:
            self.win = [9] * 2
        else:
            self.win = [win] * 2

    def ncc(self, I, J):  # noqa
        """Compute correlation."""
        # get dimension of volume

        # get convolution function
        conv_fn = tf.nn.conv2d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = [1, 1, 1, 1]
        padding = "SAME"

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):  # noqa
        """Compute loss."""
        return -self.ncc(I, J)


class PerceptualLoss:
    """Class computing perceptual loss under different hyperparameters."""

    def __init__(self, model="net-lin", net="alex"):
        self.model = model
        self.net = net

    def loss(self, y_true, y_pred):
        """Compute loss."""
        y_true_rgb = tf.concat(3 * [y_true], axis=-1)
        y_pred_rgb = tf.concat(3 * [y_pred], axis=-1)

        return lpips_tf.lpips(y_true_rgb, y_pred_rgb, model=self.model, net=self.net)


def cross_correlation(y_true, y_pred):
    """Pearson cross correlation.

    Parameters
    ----------
    y_true : K.tensor
        Tensor of shape (None, h, w, 1) representing the ground truth image.

    y_pred : K.tensor
        Tensor of shape (None, h, w, 1) representing the predicted image.

    Returns
    -------
    cc : K.Tensor
        Scalar tensor representing the 1 - correlation.
    """
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def psnr(y_true, y_pred):
    """Compute the negative of PSNR.

    Parameters
    ----------
    y_true : K.tensor
        Tensor of shape (None, h, w, 1) representing the ground truth image.

    y_pred : K.tensor
        Tensor of shape (None, h, w, 1) representing the predicted image.

    Returns
    -------
    res : K.Tensor
        Scalar tensor representing the -psnr.
    """
    return -tf.image.psnr(y_true, y_pred, 1)


def ssim(y_true, y_pred, **kwargs):
    """Compute the negative of structural similarity.

    Parameters
    ----------
    y_true : K.tensor
        Tensor of shape (None, h, w, 1) representing the ground truth image.

    y_pred : K.tensor
        Tensor of shape (None, h, w, 1) representing the predicted image.

    kwargs : dict
        Additional hyperparameters to be passed into the tensorflow implementation of ssim.

    Returns
    -------
    res : K.Tensor
        Scalar tensor representing the -ssim.

    References
    ----------
    [1] https://www.tensorflow.org/api_docs/python/tf/image/ssim

    """
    return -tf.image.ssim(y_true, y_pred, 1, **kwargs)


# DVF LOSSES
class Grad:
    """Class computing gradient loss with different hyperparameters."""

    def __init__(self, penalty="l1"):
        self.penalty = penalty

    def _diffs(self, y):
        # vol_shape = y.get_shape().as_list()[1:-1]
        # ndims = len(vol_shape)
        ndims = 2

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """Compute loss."""
        if self.penalty == "l1":
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == "l2", (
                "penalty can only be l1 or l2. Got: %s" % self.penalty
            )
            df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)


def jacobian(_, y_pred):
    """Compute loss."""
    # define layers
    first_channel = keras.layers.Lambda(lambda x: x[..., 0], name="first_channel")
    second_channel = keras.layers.Lambda(lambda x: x[..., 1], name="second_channel")

    x_pos_shift = keras.layers.Lambda(lambda x: x[:, :, 2:], name="x_pos")
    x_neg_shift = keras.layers.Lambda(lambda x: x[:, :, :-2], name="x_neg")
    y_pos_shift = keras.layers.Lambda(lambda x: x[:, 2:, :], name="y_pos")
    y_neg_shift = keras.layers.Lambda(lambda x: x[:, :-2, :], name="y_neg")

    cutoff_x = keras.layers.Lambda(lambda x: x[:, :, 1:-1], name="cutoff_x")
    cutoff_y = keras.layers.Lambda(lambda x: x[:, 1:-1, :], name="cutoff_y")

    # tensors
    delta_x = first_channel(y_pred)
    delta_y = second_channel(y_pred)

    a_11 = 1 + (-x_neg_shift(delta_x) + x_pos_shift(delta_x)) / 2
    a_12 = (-y_neg_shift(delta_x) + y_pos_shift(delta_x)) / 2
    a_21 = (-x_neg_shift(delta_y) + x_pos_shift(delta_y)) / 2
    a_22 = 1 + (-y_neg_shift(delta_y) + y_pos_shift(delta_y)) / 2

    a_11 = cutoff_y(a_11)
    a_12 = cutoff_x(a_12)
    a_21 = cutoff_y(a_21)
    a_22 = cutoff_x(a_22)

    det = keras.layers.Multiply()([a_11, a_22]) - keras.layers.Multiply()([a_12, a_21])

    n_pixels = tf.constant(np.prod(keras.backend.int_shape(det)[1:]), dtype=tf.float32)
    count_artifacts = tf.cast(
        tf.count_nonzero(tf.greater_equal(-det, 0.0), axis=(1, 2)), dtype=tf.float32
    )
    perc_artifacts = count_artifacts / n_pixels

    return keras.backend.mean(perc_artifacts)


def jacobian_distance(y_true, y_pred, norm="l2"):
    """Compute average per pixel distance between jacobians.

    Parameters
    ----------
    y_true : K.tensor
        True DVF.

    y_pred : K.tensor
        Pred DVF.

    norm : str, {'l1', 'l2'}
        Norm to use.

    """
    # define layers
    first_channel = keras.layers.Lambda(lambda x: x[..., 0], name="first_channel")
    second_channel = keras.layers.Lambda(lambda x: x[..., 1], name="second_channel")

    x_pos_shift = keras.layers.Lambda(lambda x: x[:, :, 2:], name="x_pos")
    x_neg_shift = keras.layers.Lambda(lambda x: x[:, :, :-2], name="x_neg")
    y_pos_shift = keras.layers.Lambda(lambda x: x[:, 2:, :], name="y_pos")
    y_neg_shift = keras.layers.Lambda(lambda x: x[:, :-2, :], name="y_neg")

    cutoff_x = keras.layers.Lambda(lambda x: x[:, :, 1:-1], name="cutoff_x")
    cutoff_y = keras.layers.Lambda(lambda x: x[:, 1:-1, :], name="cutoff_y")

    # tensors
    delta_x_true = first_channel(y_pred)
    delta_y_true = second_channel(y_pred)

    a_11 = 1 + (-x_neg_shift(delta_x_true) + x_pos_shift(delta_x_true)) / 2
    a_12 = (-y_neg_shift(delta_x_true) + y_pos_shift(delta_x_true)) / 2
    a_21 = (-x_neg_shift(delta_y_true) + x_pos_shift(delta_y_true)) / 2
    a_22 = 1 + (-y_neg_shift(delta_y_true) + y_pos_shift(delta_y_true)) / 2

    a_11 = cutoff_y(a_11)
    a_12 = cutoff_x(a_12)
    a_21 = cutoff_y(a_21)
    a_22 = cutoff_x(a_22)

    det_true = keras.layers.Multiply()([a_11, a_22]) - keras.layers.Multiply()(
        [a_12, a_21]
    )

    delta_x_pred = first_channel(y_pred)
    delta_y_pred = second_channel(y_pred)

    a_11_p = 1 + (-x_neg_shift(delta_x_pred) + x_pos_shift(delta_x_pred)) / 2
    a_12_p = (-y_neg_shift(delta_x_pred) + y_pos_shift(delta_x_pred)) / 2
    a_21_p = (-x_neg_shift(delta_y_pred) + x_pos_shift(delta_y_pred)) / 2
    a_22_p = 1 + (-y_neg_shift(delta_y_pred) + y_pos_shift(delta_y_pred)) / 2

    a_11_p = cutoff_y(a_11_p)
    a_12_p = cutoff_x(a_12_p)
    a_21_p = cutoff_y(a_21_p)
    a_22_p = cutoff_x(a_22_p)

    det_pred = keras.layers.Multiply()([a_11_p, a_22_p]) - keras.layers.Multiply()(
        [a_12_p, a_21_p]
    )

    if norm == "l1":
        return K.mean(K.absolute(det_true - det_pred))

    elif norm == "l2":
        return K.mean(K.square(det_true - det_pred))

    else:
        raise ValueError("Unrecognized norm {}.".format(norm))


def vector_distance(y_true, y_pred):
    """Compute loss."""
    diff = y_pred - y_true

    vector_distance_per_output = tf.reduce_mean(
        tf.sqrt(tf.abs(tf.square(diff[..., 0]) + tf.square(diff[..., 1]) + 0.001)),
        axis=0,
    )
    vector_distance_average = tf.reduce_mean(vector_distance_per_output)

    return vector_distance_average


class VDClipper:
    """Clipping class."""

    def __init__(self, value=20, power=3):
        self.value = value
        self.power = power

    def loss(self, y_true, y_pred):
        """Compute loss."""
        vd = vector_distance(y_true, y_pred)

        return tf.math.pow(vd / self.value, self.power)


def mse_po(y_true, y_pred):
    """Compute loss."""
    diff = y_pred - y_true

    vector_distance_per_output = tf.reduce_mean(
        tf.square(diff[..., 0]) + tf.square(diff[..., 1]), axis=0
    )
    vector_distance_average = tf.reduce_mean(vector_distance_per_output)

    return vector_distance_average
