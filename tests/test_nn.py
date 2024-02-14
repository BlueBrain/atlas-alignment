"""Tests focused on the nn module."""

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
import pytest
from tensorflow import keras

from atlalign.nn import supervised_global_model_factory, supervised_model_factory


class TestSupervisedModelFactory:
    """Collection of tests focused on the `supervised_model_factory`."""

    def test_default_construction(self):
        """Make sure possible to use with the default setting"""

        model = supervised_model_factory()
        assert isinstance(model, keras.Model)

    @pytest.mark.parametrize("compute_inv", [True, False])
    @pytest.mark.parametrize("use_lambda", [True, False])
    def test_use_lambda(self, use_lambda, compute_inv):
        """Make sure the `use_lambda` flag is working"""

        losses = ("mse", "mse", "mse") if compute_inv else ("mse", "mse")
        losses_weights = (1, 1, 1) if compute_inv else (1, 1)
        model = supervised_model_factory(
            losses=losses,
            losses_weights=losses_weights,
            compute_inv=compute_inv,
            use_lambda=use_lambda,
        )
        lambda_list = [x for x in model.layers if isinstance(x, keras.layers.Lambda)]
        if use_lambda:
            assert lambda_list
        else:
            assert not lambda_list

    # @pytest.mark.parametrize("compute_inv", [True, False])
    # def test_equivalence(self, compute_inv):
    #     """Make sure moving Lambda layers does not affect the results."""
    #     losses = ("mse", "mse", "mse") if compute_inv else ("mse", "mse")
    #     losses_weights = (1, 1, 1) if compute_inv else (1, 1)
    #     params = {
    #         "start_filters": (2,),
    #         "downsample_filters": (2, 3),
    #         "middle_filters": (2,),
    #         "upsample_filters": (2, 3),
    #         "end_filters": tuple(),
    #         "compute_inv": compute_inv,
    #         "losses": losses,
    #         "losses_weights": losses_weights,
    #     }
    #
    #     np.random.seed(1337)
    #     tf.random.set_seed(1337)
    #     model_with = supervised_model_factory(use_lambda=True, **params)
    #     np.random.seed(1337)
    #     tf.random.set_seed(1337)
    #     model_without = supervised_model_factory(use_lambda=False, **params)
    #     x = np.random.random((1, 320, 456, 2))
    #     pred_with = model_with.predict([x, x] if compute_inv else x)
    #     pred_without = model_without.predict([x, x] if compute_inv else x)
    #
    #     assert np.allclose(pred_with[0], pred_without[0])
    #     assert np.allclose(pred_with[1], pred_without[1])
    #     if compute_inv:
    #         assert np.allclose(pred_with[2], pred_without[2])

    def test_down_up_samples(self):
        """Make sure raises an error if downsamples and upsamples have not the same number of layers"""
        with pytest.raises(ValueError):
            supervised_model_factory(downsample_filters=(2,), upsample_filters=(2, 3))
        with pytest.raises(ValueError):
            supervised_model_factory(
                downsample_filters=(2, 2, 2, 2, 2, 2, 2),
                upsample_filters=(2, 2, 2, 2, 2, 2, 2),
            )


class TestSupervisedGlobalModelFactory:
    """Collection of tests focused on the `supervised_model_factory`."""

    def test_default_construction(self):
        """Make sure possible to use with the default setting"""

        model = supervised_global_model_factory()
        assert isinstance(model, keras.Model)

    @pytest.mark.parametrize("use_lambda", [True, False])
    def test_use_lambda(self, use_lambda):
        """Make sure the `use_lambda` flag is working"""

        model = supervised_global_model_factory(use_lambda=use_lambda)
        lambda_list = [x for x in model.layers if isinstance(x, keras.layers.Lambda)]
        if use_lambda:
            assert lambda_list
        else:
            assert not lambda_list

    def test_equivalence(self):
        """Make sure moving Lambda layers does not affect the results."""
        params = {"filters": (2, 2, 2, 2), "dense_layers": (2,)}
        np.random.seed(1337)
        model_with = supervised_global_model_factory(use_lambda=True, **params)
        np.random.seed(1337)
        model_without = supervised_global_model_factory(use_lambda=False, **params)
        x = np.random.random((1, 320, 456, 2))
        pred_with = model_with.predict(x)
        pred_without = model_without.predict(x)

        assert np.allclose(pred_with[0], pred_without[0])
        assert np.allclose(pred_with[1], pred_without[1])
