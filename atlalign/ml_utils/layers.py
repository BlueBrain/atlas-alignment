"""Custom keras layers."""

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
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Lambda,
    MaxPool2D,
    ReLU,
    Reshape,
)


def K_meshgrid(x, y):
    """Generate meshgrid.

    Notes
    -----
    Not part of Keras backend convertor.

    """
    return tf.meshgrid(x, y)


def get_initial_weights(previous_layer_size):
    """Initialize affine matrix to identity transformation.

    Notes
    -----
    It makes a lot of sense to set the behavior before training to learning an identity
    transformation.

    Parameters
    ----------
    previous_layer_size : int
        Size of the previous dense layer.

    Returns
    -------
    initial_weights : list
        A list of 2 numpy ndarrays:
            1) weight matrix, shape (previous_layer_size, 6)
            2) biases, shape (6,)

    """
    b = np.zeros((2, 3), dtype="float32")
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((previous_layer_size, 6), dtype="float32")
    weights = [W, b.flatten()]

    return weights


class BilinearInterpolation_(Layer):
    """Perform bilinear interpolation as a Keras layer.

    Notes
    -----
    Currently only works with tensorflow backend since we need to use tf.meshgrid.

    """

    def __init__(self, **kwargs):  # noqa
        super(BilinearInterpolation_, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        """Compute output shape.

        Notes
        -----
        This is a mandatory method to implement for custom Keras layers.

        Parameters
        ----------
        input_shapes : list
            A list of the shape of images to be warped and the dvfs respectively.
            - images to be warped : (batch_size, h, w, 1)
            - dvfs : (batch_size, h, w, 2)

        Returns
        -------
        None
            Batch size

        height : int
            Height of the image.

        width : int
            Width of the image.

        n_channels : int
            Number of channels, should be 1.

        """
        height, width, n_channels = input_shapes[0][1:]

        return None, height, width, n_channels

    def call(self, tensors, mask=None):
        """Perform forward pass.

        Parameters
        ----------
        tensors : list
            A list of two elements - images to we warped and the displacement vector fields. Their shapes are the
            following

                - images to be warped : (batch_size, h, w, 1)
                - dvfs : (batch_size, h, w, 2)

        mask : None

        Returns
        -------
        output : K.Tensor
            Warped images with the provided dvfs. Shape (batch_size, h, w, 1).

        """
        imgs, dvfs = tensors

        # dimensions as tensors
        batch_size = K.shape(imgs)[0]
        height = K.shape(imgs)[1]
        width = K.shape(imgs)[2]
        num_channels = K.shape(imgs)[3]

        sampled_grids = self._create_grid(dvfs, batch_size, height, width)
        interpolated_images = self._interpolate(imgs, sampled_grids)
        new_shape = (batch_size, height, width, num_channels)
        interpolated_images = K.reshape(interpolated_images, new_shape)

        return interpolated_images

    def _create_grid(self, dvfs, batch_size, height, width):
        """Create a regular grid.

        Returns
        -------
        sample_grid : K.Tensor
            Regular grid.

        """
        # making a single regular grid
        x_linspace = K.arange(0, width, dtype="int32")
        y_linspace = K.arange(0, height, dtype="int32")

        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)

        x_coordinates_f = K.flatten(x_coordinates)
        y_coordinates_f = K.flatten(y_coordinates)

        grid = K.expand_dims(K.stack([x_coordinates_f, y_coordinates_f], 0), 0)
        grids = K.tile(grid, K.stack([batch_size, 1, 1]))

        regular_grid = K.cast(
            K.reshape(grids, (batch_size, 2, height * width)), dtype="float32"
        )

        dvfs_ = K.permute_dimensions(dvfs, [0, 3, 1, 2])
        sampled_grid = K.reshape(dvfs_, (batch_size, 2, -1)) + regular_grid

        return sampled_grid

    def _interpolate(self, image, sampled_grids):
        """Perform the actual interpolation.

        Parameters
        ----------
        image : K.Tensor

        sampled_grids : K.Tensor

        Returns
        -------
        warped_image : K.Tensor

        """
        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.flatten(sampled_grids[:, 0:1, :])
        y = K.flatten(sampled_grids[:, 1:2, :])

        # x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        # y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, "int32")
        x1 = x0 + 1
        y0 = K.cast(y, "int32")
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = (
            K.int_shape(image)[1] * K.int_shape(image)[2]
        )  # must be an integer
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype="float32")
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, "float32")
        x1 = K.cast(x1, "float32")
        y0 = K.cast(y0, "float32")
        y1 = K.cast(y1, "float32")

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d

        return values_a + values_b + values_c + values_d


class BilinearInterpolation(Layer):
    """Implementation using tf.contrib."""

    def __init__(self, **kwargs):  # noqa
        super(BilinearInterpolation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        """Compute output shape.

        Notes
        -----
        This is a mandatory method to implement for custom Keras layers.

        Parameters
        ----------
        input_shapes : list
            A list of the shape of images to be warped and the dvfs respectively.
            - images to be warped : (batch_size, h, w, 1)
            - dvfs : (batch_size, h, w, 2)

        Returns
        -------
        None
            Batch size

        height : int
            Height of the image.

        width : int
            Width of the image.

        n_channels : int
            Number of channels, should be 1.

        """
        height, width, n_channels = input_shapes[0][1:]

        return None, height, width, n_channels

    def call(self, tensors, mask=None):
        """Perform forward pass.

        Parameters
        ----------
        tensors : list
            A list of two elements - images to we warped and the displacement vector fields. Their shapes are the
            following

                * images to be warped : (batch_size, h, w, 1)
                * dvfs : (batch_size, h, w, 2)

        mask : None

        Returns
        -------
        output : K.Tensor
            Warped images with the provided dvfs. Shape (batch_size, h, w, 1).

        """
        imgs, dvfs = tensors

        # dimensions as tensors
        batch_size = K.shape(imgs)[0]
        height = K.shape(imgs)[1]
        width = K.shape(imgs)[2]

        x_linspace = K.arange(0, width, dtype="int32")
        y_linspace = K.arange(0, height, dtype="int32")

        x_coordinates, y_coordinates = K_meshgrid(
            x_linspace, y_linspace
        )  # 2 x (height, width)
        grid = K.tile(
            K.expand_dims(K.stack((x_coordinates, y_coordinates), axis=2), 0),
            (batch_size, 1, 1, 1),
        )
        grid = K.cast(grid, dtype="float32")

        f_x_f_y = grid + dvfs

        output = tf.contrib.resampler.resampler(imgs, f_x_f_y)

        return output


class DVFComposition(Layer):
    """Composition of 2 displacement vector fields.

    Notes
    -----
    Computes the DVF of f ∘ g

    """

    def __init__(self, **kwargs):  # noqa
        super(DVFComposition, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        """Compute output shape.

        Parameters
        ----------
        input_shapes : list
            A list of the shape of th 2 dvfs respectively.
            - dvfs_outer: (batch_size, h, w, 2)
            - dvfs_inner : (batch_size, h, w, 2)


        Returns
        -------
        None
            Batch size

        height : int
            Height of the image.

        width : int
            Width of the image.

        n_dim : int
            2

        """
        height, width = input_shapes[0][1:3]

        return None, height, width, 2

    def call(self, tensors, mask=None):
        """Composition.

        Parameters
        ----------
        tensors : list
            A list of two elements - outer and inner displacement vector field. Their shapes are identical and equal
            to (batch_size, h, w, 2).

        Returns
        -------
        output : K.tensor
            A displacement vector field of the composite transformation of shape (batch_size, h, w, 2).

        """
        dvfs_outer, dvfs_inner = tensors

        batch_size = K.shape(dvfs_outer)[0]
        height = K.shape(dvfs_outer)[1]
        width = K.shape(dvfs_outer)[2]

        x_linspace = K.arange(0, width, dtype="int32")
        y_linspace = K.arange(0, height, dtype="int32")

        x_coordinates, y_coordinates = K_meshgrid(
            x_linspace, y_linspace
        )  # 2 x (height, width)
        grid = K.tile(
            K.expand_dims(K.stack((x_coordinates, y_coordinates), axis=2), 0),
            (batch_size, 1, 1, 1),
        )
        grid = K.cast(grid, dtype="float32")  # (batch_size, h, w, 2)

        delta_x = (
            BilinearInterpolation()(
                [dvfs_outer[:, :, :, :1] + grid[:, :, :, :1], dvfs_inner]
            )
            - grid[:, :, :, :1]
        )
        delta_y = (
            BilinearInterpolation()(
                [dvfs_outer[:, :, :, 1:] + grid[:, :, :, 1:], dvfs_inner]
            )
            - grid[:, :, :, 1:]
        )

        dvfs_composition = K.concatenate((delta_x, delta_y), axis=3)
        return dvfs_composition


class Affine2DVF(Layer):
    """Given an affine transformation matrix (2 x 3) generate the corresponding DVF."""

    def __init__(self, shape, **kwargs):
        """Construct.

        Parameters
        ----------
        shape : tuple
            2 element tuple (height, width).

        kwargs : dict
            Whatever keywords arguments passed intho the ``Layer`` constructor.

        """
        if not len(shape) == 2:
            raise ValueError("Height and width only need to be passed.")

        self.shape = shape

        super(Affine2DVF, self).__init__(**kwargs)

    def compute_output_shape(self, *args):
        """Compute the output shape.

        Parameters
        ----------
        args : list
            Whatever positional arguments. We already know the expected shape from the `self.shape` passed
            into the constructor.

        Returns
        -------
        batch_size : int
            Always None since we want to allow for variable batch_size.

        height : int
            Height of the image.

        width : int
            Width of the image.

        n_channels : int
            Always equal to 2 because it represents delta_x and delta_y.

        """
        return None, self.shape[0], self.shape[1], 2

    def call(self, a, mask=None):
        """Turn a batch of affine matrices into the corresponding batch of displacement vector fields.

        Parameters
        ----------
        a : K.Tensor
            A tensor of shape (batch_size, 2, 3) representing an affine transformation for each sample.

        Returns
        -------
        dvfs : K.Tensor
            A tensor of shape (batch_size, h, w, 2) representing the corresponding displacement vector fields.

        """
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        # regular_grids = self._make_regular_grids(batch_size, *output_size)

        batch_size = K.shape(a)[0]
        height, width = self.shape

        x_linspace = K.arange(0, width, dtype="int32")
        y_linspace = K.arange(0, height, dtype="int32")
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate(
            [x_coordinates, y_coordinates, ones], 0
        )  # also with homogeneous coordinates

        # Add batch_dimension
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        regular_grids = K.cast(
            K.reshape(grids, (batch_size, 3, height * width)), dtype="float32"
        )

        # Homogeneous coordinates
        a_homog = K.map_fn(lambda x: tf.concat((x, [[0, 0, 1]]), axis=0), a)

        # Compute
        coords = K.batch_dot(a_homog, regular_grids)
        coords_delta = coords - regular_grids

        dvfs = K.stack(
            (
                K.reshape((coords_delta[:, 0, :]), (batch_size, height, width)),
                K.reshape((coords_delta[:, 1, :]), (batch_size, height, width)),
            ),
            axis=3,
        )
        return dvfs

    def get_config(self):
        """Allow for model to be loaded correctly.

        We just need to add to the config dictionary all the custom constructor parameters.

        """
        base_config = super().get_config()
        base_config["shape"] = self.shape

        return base_config


class NoOp(Layer):
    """No operation layer."""

    def call(self, x):
        """Perform forward pass."""
        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape


class ExtractMoving(Layer):
    """Extract the moving image from the input."""

    def call(self, x):
        """Perform forward pass."""
        return x[:, :, :, 1:]

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return (*input_shape[0:3], 1)


def block_stn(
    images,
    filters=(16, 32, 64),
    dense_layers=(50,),
    block_name="STN",
    return_dvfs=True,
    return_inverse=True,
    include_bn_dense=False,
    include_bn_conv=False,
):
    """Spatial transformer network block.

    Parameters
    ----------
    images : K.tensor
        Tensor of shape (batch_size, h, w, 2) representing the batch of images to be warped. The first channel
         represents the atlas image and the second one is the moving image.

    filters : tuple
        A tuple whose length represents the number of convolutional layers and each element represents the number
        of filters of that layer. Make sure that h % (2 ** len(filters)) == 0 and w % (2 ** len(filters)) == 0 because
        of max pooling.

    dense_layers : tuple
        A tuple whose length represents the number of dense layers after the convolutional layers and each element
        represents the number of nodes. Note a final dense layer with 8 nodes (number of parameters in the
        affine transformation) is added automatically everytime.

    block_name : str
        Name of the block.

    return_dvfs : bool
        If True then also returns the `dsvf` tensor then can later be used for composition.

    return_inverse : bool
        If True, then warping applied to the second channel (moving image), otherwise
        the first image (atlas).

    include_bn_dense : bool
        If True, batch normalization is performed after each dense layer layer.

    include_bn_conv : bool
        If True, batch normalization is performed after each convolutional layer.

    Returns
    -------
    warped_images : K.tensor
        Tensor of shape (batch_size, h, w, 1) representing the batch of warped images with the predicted affine
         transformation. If `return_inverse` True then this is the moving image. Else the reference.

     dvfs : K.tensor
        Tensor of shape (batch_size, h, w, 2) representing the batch of delta_x and delta_y displacements. Only returned
        if `return_dvfs` is True.

    """
    if not dense_layers:
        raise ValueError("There needs to be at least 1 hidden dense layer.")

    with K.name_scope(block_name):
        # h, w = K.shape(images)[1], K.shape(images)[2]
        h, w = K.int_shape(images)[1], K.int_shape(images)[2]

        x = images

        # Convolutions
        for f in filters:
            x = Conv2D(filters=f, strides=(1, 1), kernel_size=(3, 3), padding="same")(x)
            if include_bn_conv:
                x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPool2D(pool_size=(2, 2))(x)

        # Dense
        x = Flatten()(x)

        for d in dense_layers:
            x = Dense(units=d)(x)
            if include_bn_dense:
                x = BatchNormalization()(x)
            x = ReLU()(x)

        a = Dense(6, weights=get_initial_weights(dense_layers[-1]))(x)

        a = Reshape((2, 3))(a)

        dvfs = Affine2DVF(shape=(h, w))(a)

        extract_layer = Lambda(
            lambda x: (x[:, :, :, 1:] if return_inverse else x[:, :, :, :1]),
            name="extract",
        )

        warped_images = BilinearInterpolation()([extract_layer(images), dvfs])

        if return_dvfs:
            return warped_images, dvfs

        else:
            return warped_images
