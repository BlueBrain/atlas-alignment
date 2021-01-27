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

import matplotlib.pyplot as plt
import numpy as np
import pytest
from skimage.draw import rectangle

import keras.backend as K
from keras.layers import Input
from keras.models import Model, load_model, save_model
import tensorflow as tf

from atlalign.base import DisplacementField
from atlalign.ml_utils import (
    Affine2DVF,
    BilinearInterpolation,
    DVFComposition,
    block_stn,
)

from atlalign.ml_utils.layers import BilinearInterpolation_


class TestBilinearInterpolation:
    """
    Notes
    -----
    Assumes using tensorflow.
    """

    @pytest.mark.parametrize("layer_name", ["native"])
    @pytest.mark.parametrize("batch_size", [1, 2, 10])
    def test_possible_warp_rectangle(self, batch_size, layer_name):
        """Affine warping of an image with all edges being black.

        Notes
        -----
        For forward pass not necessary to compile a model.
        See https://github.com/keras-team/keras/issues/3074#issuecomment-228604501

        Additionally the manually implemented interpolation layer ``BilinearInterpolation_`` struggles with
        images with nonzero last rows and columns.

        """

        allow_visualize = False

        (h, w) = 356 // 2, 420 // 2

        imgs = Input((h, w, 1))
        dvfs = Input((h, w, 2))

        layer = (
            BilinearInterpolation()
            if (layer_name == "native")
            else BilinearInterpolation_()
        )
        x = layer([imgs, dvfs])

        assert isinstance(x, tf.Tensor)
        assert x.shape.ndims == 4

        model = Model(inputs=[imgs, dvfs], outputs=x)

        # Draw a circle
        rr, cc = rectangle((h // 4, w // 4), end=((3 * h) // 4, (3 * w) // 4))

        img_a_single = np.zeros((h, w))
        img_a_single[rr, cc] = 0.5

        imgs_a = np.tile(
            img_a_single[np.newaxis, ..., np.newaxis], (batch_size, 1, 1, 1)
        )
        dvfs_a = np.zeros((batch_size, h, w, 2))  # to be filled
        imgs_warped_true = np.zeros((batch_size, h, w, 1))  # to be filled

        for i in range(batch_size):
            df = DisplacementField.generate(
                (h, w),
                approach="affine_simple",
                translation_x=2,
                rotation=(i + 1) * 3.14 / (batch_size + 1),  # just avoid 180 and 360
                translation_y=2,
            )

            imgs_warped_true[i, :, :, 0] = df.warp(
                img_a_single, interpolation="linear", border_mode="replicate"
            )
            dvfs_a[i, :, :, 0] = df.delta_x
            dvfs_a[i, :, :, 1] = df.delta_y

        # run
        imgs_warped_pred = model.predict([imgs_a, dvfs_a])

        # visualize

        if allow_visualize:
            for i in range(batch_size):
                _, (ax_orig, ax_open, ax_tf) = plt.subplots(1, 3, figsize=(15, 15))
                ax_orig.imshow(np.squeeze(img_a_single))
                ax_open.imshow(np.squeeze(imgs_warped_true[i, ..., 0]))
                ax_tf.imshow(np.squeeze(imgs_warped_pred[i, ..., 0]))
                plt.show()

        assert imgs_warped_pred.shape == (batch_size, h, w, 1)
        assert np.allclose(imgs_warped_true, imgs_warped_pred, atol=0.02, rtol=0)

    @pytest.mark.parametrize(
        "layer_name", ["native", pytest.param("manual", marks=pytest.mark.xfail)]
    )
    @pytest.mark.parametrize("dvf_type", ["identity", "affine"])
    def test_possible_warp_real_image(self, img_grayscale_float, dvf_type, layer_name):
        """Warping of an image with non black edges.

        Notes
        -----
        For forward pass not necessary to compile a model.
        See https://github.com/keras-team/keras/issues/3074#issuecomment-228604501

        Additionally the manually implemented interpolation layer ``BilinearInterpolation_`` struggles with
        images with nonzero last rows and columns.

        """

        allow_visualize = False
        (h, w) = img_grayscale_float.shape
        img_grayscale_float_ = img_grayscale_float[np.newaxis, :, :, np.newaxis]

        imgs = Input((h, w, 1))
        dvfs = Input((h, w, 2))

        layer = (
            BilinearInterpolation()
            if layer_name == "native"
            else BilinearInterpolation_()
        )
        x = layer([imgs, dvfs])

        assert isinstance(x, tf.Tensor)
        assert x.shape.ndims == 4

        model = Model(inputs=[imgs, dvfs], outputs=x)
        model.compile(optimizer="adam", loss="mse")

        if dvf_type == "identity":
            df = DisplacementField.generate((h, w), approach="identity")
        elif dvf_type == "affine":
            df = DisplacementField.generate(
                (h, w), approach="affine_simple", rotation=0.25
            )

        else:
            raise ValueError("Unrecognized dvf_type {}".format(dvf_type))

        img_warped_true = df.warp(
            img_grayscale_float, interpolation="linear", border_mode="constant", c=0
        )[
            np.newaxis, :, :, np.newaxis
        ]  # noqa

        dvfs_a = np.zeros((h, w, 2))
        dvfs_a[..., 0] = df.delta_x
        dvfs_a[..., 1] = df.delta_y

        dvfs_a = dvfs_a[np.newaxis, ...]
        img_warped_pred = model.predict([img_grayscale_float_, dvfs_a])

        if allow_visualize:
            _, (ax_orig, ax_open, ax_tf) = plt.subplots(1, 3, figsize=(15, 15))
            ax_orig.imshow(np.squeeze(img_grayscale_float))
            ax_open.imshow(np.squeeze(img_warped_true))
            ax_tf.imshow(np.squeeze(img_warped_pred))
            plt.show()

        assert np.allclose(img_warped_true, img_warped_pred, atol=0.02, rtol=0)


class TestDVFComposition:
    """A collection of tests focused on the `DVFComposition` layer."""

    @pytest.mark.parametrize("batch_size", [1, 2, 10])
    def test_composition(self, batch_size, img_grayscale_float):
        """Composition of two displacement fields.

        Notes
        -----
        For some reasons the approximation differences are way more pronounced here, need to set higher tolerance.
        For forward pass not necessary to compile a model.
        See https://github.com/keras-team/keras/issues/3074#issuecomment-228604501

        """
        allow_visualize = False

        random_state = 8

        (h, w) = 356 // 2, 420 // 2

        dvfs_outer = Input((h, w, 2))
        dvfs_inner = Input((h, w, 2))

        layer = DVFComposition()
        x = layer([dvfs_outer, dvfs_inner])

        assert isinstance(x, tf.Tensor)
        assert x.shape.ndims == 4

        model = Model(inputs=[dvfs_outer, dvfs_inner], outputs=x)

        dvfs_a_inner = np.zeros((batch_size, h, w, 2))  # to be filled
        dvfs_a_outer = np.zeros((batch_size, h, w, 2))  # to be filled
        dvfs_a_composition_true = np.zeros((batch_size, h, w, 2))  # to be filled

        for i in range(batch_size):
            df_outer = DisplacementField.generate(
                (h, w),
                approach="paper",
                n_pixels=50,
                kernel_sigma=20,
                random_state=random_state,
            )
            df_inner = DisplacementField.generate(
                (h, w),
                approach="affine_simple",
                translation_x=20,
                rotation=(i + 1) * 3.14 / (batch_size + 1),  # just avoid 180 and 360
                translation_y=2,
            )

            df_composition = df_outer(
                df_inner, interpolation="linear", border_mode="constant", c=0
            )

            dvfs_a_inner[i, :, :, 0] = df_inner.delta_x
            dvfs_a_inner[i, :, :, 1] = df_inner.delta_y
            dvfs_a_outer[i, :, :, 0] = df_outer.delta_x
            dvfs_a_outer[i, :, :, 1] = df_outer.delta_y
            dvfs_a_composition_true[i, :, :, 0] = df_composition.delta_x
            dvfs_a_composition_true[i, :, :, 1] = df_composition.delta_y

        # run
        dvfs_a_composition_pred = model.predict([dvfs_a_outer, dvfs_a_inner])

        if allow_visualize:
            # Warped image
            rr, cc = rectangle((h // 4, w // 4), end=((3 * h) // 4, (3 * w) // 4))
            img_a_single = np.zeros((h, w))
            img_a_single[rr, cc] = 0.5

            df_true = DisplacementField(
                dvfs_a_composition_true[0, ..., 0], dvfs_a_composition_true[0, ..., 1]
            )
            df_pred = DisplacementField(
                dvfs_a_composition_pred[0, ..., 0], dvfs_a_composition_pred[0, ..., 1]
            )

            _, (ax_warp_true, ax_warp_pred) = plt.subplots(1, 2, figsize=(15, 15))
            ax_warp_true.imshow(df_true.warp(img_a_single))
            ax_warp_pred.imshow(df_pred.warp(img_a_single))

            # actual transform
            _, ((ax_true_dx, ax_pred_dx), (ax_true_dy, ax_pred_dy)) = plt.subplots(
                2, 2, figsize=(14, 14)
            )
            ax_pred_dx.imshow(dvfs_a_composition_pred[0, ..., 0])
            ax_true_dx.imshow(dvfs_a_composition_true[0, ..., 0])

            ax_pred_dy.imshow(dvfs_a_composition_pred[0, ..., 1])
            ax_true_dy.imshow(dvfs_a_composition_true[0, ..., 1])

            plt.show()

        assert dvfs_a_composition_pred.shape == (batch_size, h, w, 2)
        assert abs(dvfs_a_composition_pred - dvfs_a_composition_true).mean() < 0.1
        assert np.allclose(
            dvfs_a_composition_pred, dvfs_a_composition_true, atol=4, rtol=0
        )


class TestAffine2DVF:
    """A collection of tests focused on the `Affine2DVF` layer."""

    @pytest.mark.parametrize(
        "a",
        [
            np.array([[1, 0, 0], [0, 1, 0]]),  # identity
            np.array([[1.1, 0, 0], [0, 1.1, 0]]),  # zoom
            np.array([[0, -1, 0], [1, 0, 0]]),  # rotation 90
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 2, 10])
    def test_basic(self, batch_size, a):
        """Basic functionality"""

        (h, w) = 356 // 2, 420 // 2

        a_tensor = Input((2, 3))
        layer = Affine2DVF((h, w))
        x = layer(a_tensor)

        assert isinstance(x, tf.Tensor)
        assert x.shape.ndims == 4
        assert K.int_shape(x) == (None, h, w, 2)

        model = Model(inputs=a_tensor, outputs=x)

        # Initialize
        a_input = np.zeros((batch_size, 2, 3))
        dvfs_true = np.zeros((batch_size, h, w, 2))

        for i in range(batch_size):
            # add random translation for each sample
            a_input[i] = a + np.hstack(
                (np.zeros((2, 2)), np.random.uniform(-10, 10, size=(2, 1)))
            )
            df = DisplacementField.generate(
                (h, w), approach="affine", matrix=a_input[i]
            )

            dvfs_true[i, :, :, 0] = df.delta_x
            dvfs_true[i, :, :, 1] = df.delta_y

        # run
        dvfs_pred = model.predict(a_input)

        assert dvfs_pred.shape == (batch_size, h, w, 2)
        assert np.allclose(dvfs_true, dvfs_pred, atol=0.01)

    def test_possible_save_load(self, tmpdir):
        """Test that possible to save and load"""

        (h, w) = 356 // 2, 420 // 2

        a_tensor = Input((2, 3))
        layer = Affine2DVF((h, w))
        x = layer(a_tensor)

        assert isinstance(x, tf.Tensor)
        assert x.shape.ndims == 4
        assert K.int_shape(x) == (None, h, w, 2)

        model = Model(inputs=a_tensor, outputs=x)

        model_str = "fake_model_Affine2DVF.h5"

        save_model(model, str(tmpdir) + model_str)

        loaded_model = load_model(
            str(tmpdir) + model_str, custom_objects={"Affine2DVF": Affine2DVF}
        )

        assert len(model.layers) == len(loaded_model.layers)


class TestBlockSTN:
    """A collection of tests focuses on the `block_stn` function."""

    @pytest.mark.parametrize("include_bn_conv", [True, False])
    @pytest.mark.parametrize("include_bn_dense", [True, False])
    def test_n_layers(self, include_bn_dense, include_bn_conv):
        """Counting expected number of layers."""

        h, w = 64, 128
        filters = (1, 2, 3, 4)
        dense_layers = (20,)

        images = Input((h, w, 2))

        warped_images, dvfs = block_stn(
            images,
            filters=filters,
            dense_layers=dense_layers,
            return_dvfs=True,
            include_bn_dense=include_bn_dense,
            include_bn_conv=include_bn_conv,
        )

        model = Model(inputs=images, outputs=[warped_images, dvfs])

        assert K.int_shape(warped_images)[-1] == 1

        # checks
        n_inputs = 1
        n_flattens = 1
        n_reshapes = 1
        n_output_dense = 1
        n_aff2dvf = 1
        n_extract = 1
        n_bilinear = 1

        n_layers = (
            n_inputs
            + (3 + (1 if include_bn_conv else 0)) * len(filters)
            + n_flattens
            + (2 + (1 if include_bn_dense else 0)) * len(dense_layers)
            + n_output_dense
            + n_reshapes
            + n_aff2dvf
            + n_extract
            + n_bilinear
        )

        assert len(model.layers) == n_layers

    def test_correct_dvfs(self):
        """Make sure that dvfs has a deterministic shape and can be used as output and fit works."""

        shape = (h, w, c) = 64, 128, 2
        filters = (1, 2, 3, 4)
        dense_layers = (20,)

        images = Input(shape)

        warped_images, dvfs = block_stn(
            images, filters=filters, dense_layers=dense_layers, return_dvfs=True
        )

        assert K.int_shape(warped_images)[1:] == (h, w, 1)
        assert K.int_shape(dvfs)[1:] == shape  # needs to be fixed

    @pytest.mark.slow
    def test_possible_to_save_load(self, tmpdir):
        """Make sure possible to save and load.

        Loading keras model with custom layers is slightly tricky.
        """

        h, w = 16, 32
        filters = (1, 2)
        dense_layers = (1,)

        images = Input((h, w, 2))

        warped_images, dvfs = block_stn(
            images, filters=filters, dense_layers=dense_layers, return_dvfs=True
        )

        model = Model(inputs=images, outputs=[warped_images, dvfs])

        model_str = "fake_model.h5"

        save_model(model, str(tmpdir) + model_str)

        loaded_model = load_model(
            str(tmpdir) + model_str,
            custom_objects={
                "Affine2DVF": Affine2DVF,
                "BilinearInterpolation": BilinearInterpolation,
                "DVFComposition": DVFComposition,
            },
        )

        assert len(model.layers) == len(loaded_model.layers)
