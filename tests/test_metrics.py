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

from functools import partial

import numpy as np
import pytest

from atlalign.base import DisplacementField
from atlalign.metrics import (
    _compute_weights_kp,
    _euclidean_distance_kp,
    angular_error_of,
    correlation_combined,
    cross_correlation_img,
    dice_score,
    improvement_kp,
    iou_score,
    mae_combined,
    mae_img,
    mi_img,
    mse_combined,
    mse_img,
    perceptual_loss_img,
    psnr_img,
    r2_combined,
    rtre_kp,
    ssmi_img,
    tre_kp,
    vector_distance_combined,
)

ALL_DVF_METRICS = {
    "angular_error": angular_error_of,
    "correlation_combined": correlation_combined,
    "mae_combined": mae_combined,
    "mse_combined": mse_combined,
    "r2_combined": r2_combined,
    "vector_distance": vector_distance_combined,
}

ALL_ANNOTATION_METRICS = {"dice_score": dice_score, "iou_score": iou_score}

ALL_IMG_METRICS = {
    "mae": mae_img,
    "mse": mse_img,
    "psnr": psnr_img,
    "ssmi": ssmi_img,
    "mi_m": partial(mi_img, metric_type="MattesMutualInformation"),
    "mi_h": partial(mi_img, metric_type="JointHistogramMutualInformation"),
    "cross_correlation": cross_correlation_img,
    "perceptual_loss_net_vgg": partial(perceptual_loss_img, model="net", net="vgg"),
    "perceptual_loss_net_alex": partial(perceptual_loss_img, model="net", net="alex"),
    "perceptual_loss_netlin_vgg": partial(
        perceptual_loss_img, model="net-lin", net="vgg"
    ),
    "perceptual_loss_netlin_alex": partial(
        perceptual_loss_img, model="net-lin", net="alex"
    ),
}


class TestDVFMetrics:
    """A collection of tests focused on the `r2_combined` metric."""

    @pytest.mark.parametrize("metric_name", list(ALL_DVF_METRICS.keys()))
    def test_wrong_inputs_1(self, metric_name):
        """Inputs either np.ndarray or list of DisplacementField instances."""

        y_true = "1111"
        y_pred = np.zeros((2, 10, 12, 2))

        with pytest.raises(TypeError):
            ALL_DVF_METRICS[metric_name](y_true, y_pred)

    @pytest.mark.parametrize("metric_name", list(ALL_DVF_METRICS.keys()))
    def test_wrong_inputs_2(self, metric_name):
        """Wrong shapes of np.ndarray."""

        y_true = np.zeros((2, 10, 12, 2))
        y_pred = np.zeros((2, 10, 12, 3))

        with pytest.raises(ValueError):
            ALL_DVF_METRICS[metric_name](y_true, y_pred)

    @pytest.mark.parametrize("metric_name", list(ALL_DVF_METRICS.keys()))
    def test_wrong_inputs_3(self, metric_name):
        """Inconsistent shapes."""

        y_true = np.zeros((2, 20, 12, 2))
        y_pred = np.zeros((2, 10, 12, 2))

        with pytest.raises(ValueError):
            ALL_DVF_METRICS[metric_name](y_true, y_pred)

    @pytest.mark.parametrize("metric_name", list(ALL_DVF_METRICS.keys()))
    def test_wrong_inputs_4(self, metric_name):
        """Wrong dimensions."""

        y_true = np.zeros((2, 20, 12, 2, 2))
        y_pred = np.zeros((2, 10, 12, 2, 2))

        with pytest.raises(ValueError):
            ALL_DVF_METRICS[metric_name](y_true, y_pred)

    @pytest.mark.parametrize("metric_name", list(ALL_DVF_METRICS.keys()))
    def test_wrong_inputs_5(self, metric_name):
        """Wrong input type"""

        y_true = np.zeros((2, 20, 12, 2))
        y_pred = [DisplacementField.generate((20, 12), approach="identity"), "faaake"]

        with pytest.raises(TypeError):
            ALL_DVF_METRICS[metric_name](y_true, y_pred)

    @pytest.mark.parametrize("metric_name", list(ALL_DVF_METRICS.keys()))
    @pytest.mark.parametrize("df_is", ["true", "pred", "both"])
    def test_conversion_possible(self, df_is, metric_name):
        """List of DisplacementField instances to ndarray."""

        custom_list = [
            DisplacementField.generate((20, 12), approach="identity") for _ in range(2)
        ]

        y_true = np.zeros((2, 20, 12, 2)) if df_is == "pred" else custom_list
        y_pred = np.zeros((2, 20, 12, 2)) if df_is == "true" else custom_list

        _, _ = ALL_DVF_METRICS[metric_name](y_true, y_pred)

    @pytest.mark.parametrize("metric_name", list(ALL_DVF_METRICS.keys()))
    def test_perfect_prediction(self, metric_name):
        """Perfect prediction gives"""

        y_true = np.random.random((5, 10, 11, 2))
        y_pred = y_true

        metric_average, metric_per_output = ALL_DVF_METRICS[metric_name](y_true, y_pred)

        if metric_name in {"correlation_combined", "r2_combined"}:
            assert metric_average == 1
            assert np.allclose(metric_per_output, np.ones(y_true.shape[1:]))

        elif metric_name in {"mae_combined", "mse_combined"}:
            assert metric_average == 0
            assert np.allclose(metric_per_output, np.zeros(y_true.shape[1:]))

        elif metric_name in {"angular_error", "vector_distance"}:
            assert metric_average == pytest.approx(0, abs=1e-6)
            assert np.allclose(
                metric_per_output, np.zeros(y_true.shape[1:-1]), atol=1e-6
            )

        else:
            raise ValueError("Unrecognized metric name {}".format(metric_name))

    @pytest.mark.parametrize("metric_name", list(ALL_DVF_METRICS.keys()))
    def test_average_prediction(self, metric_name):
        """Average prediction."""

        y_true = np.random.random((5, 10, 11, 2)) * 10
        y_pred = np.tile(
            y_true.mean(axis=0), (5, 1, 1, 1)
        )  # prediction = average of observations

        metric_average, metric_per_output = ALL_DVF_METRICS[metric_name](y_true, y_pred)

        if metric_name == "r2_combined":
            assert metric_average == 0
            assert np.allclose(metric_per_output, np.zeros_like(y_true))

        elif metric_name == "angular_error":
            # not straightforward
            pytest.skip()

        elif metric_name == "correlation_combined":
            assert np.isnan(
                metric_average
            )  # y_pred has zero variance..division by zero

        elif metric_name == "mae_combined":
            correct_metric_per_output = np.average(abs(y_true - y_pred), axis=0)

            assert abs(metric_average - correct_metric_per_output.mean()) < 1e-6
            assert np.allclose(metric_per_output, correct_metric_per_output)

        elif metric_name == "mse_combined":
            # just variance
            correct_metric_per_output = np.var(y_true, axis=0)

            assert abs(metric_average - correct_metric_per_output.mean()) < 1e-6
            assert np.allclose(metric_per_output, correct_metric_per_output)

        elif metric_name == "vector_distance":
            correct_metric_per_output = np.zeros(y_true.shape[1:-1])

            # Just do it in a stupid way
            for r in range(correct_metric_per_output.shape[0]):
                for c in range(correct_metric_per_output.shape[1]):
                    vd_rc = []
                    for i in range(len(y_true)):
                        temp = np.sqrt(
                            (y_pred[i, r, c, 0] - y_true[i, r, c, 0]) ** 2
                            + (y_pred[i, r, c, 1] - y_true[i, r, c, 1]) ** 2
                        )

                        vd_rc.append(temp)
                    correct_metric_per_output[r, c] = np.mean(vd_rc)

            assert abs(metric_average - correct_metric_per_output.mean()) < 1e-6
            assert np.allclose(metric_per_output, correct_metric_per_output)

        else:
            raise ValueError("Unrecognized metric name {}".format(metric_name))


class TestAnnotationMetrics:
    """A collection of tests focused on the annotation metrics."""

    @pytest.mark.parametrize("metric_name", list(ALL_ANNOTATION_METRICS.keys()))
    def test_wrong_input_1(self, metric_name):
        """Incorrect type"""

        y_true = "ast"
        y_pred = np.zeros((10, 2, 3))

        with pytest.raises(TypeError):
            ALL_ANNOTATION_METRICS[metric_name](y_true, y_pred, k=0)

    @pytest.mark.parametrize("metric_name", list(ALL_ANNOTATION_METRICS.keys()))
    def test_wrong_input_2(self, metric_name):
        """Wrong dimensions"""

        y_true = np.zeros((10, 2, 3, 3))
        y_pred = np.zeros((10, 2, 3, 3))

        with pytest.raises(ValueError):
            ALL_ANNOTATION_METRICS[metric_name](y_true, y_pred, k=0)

    @pytest.mark.parametrize("metric_name", list(ALL_ANNOTATION_METRICS.keys()))
    def test_wrong_input_3(self, metric_name):
        """Different shapes."""

        y_true = np.zeros((10, 2, 3))
        y_pred = np.zeros((10, 2, 4))

        with pytest.raises(ValueError):
            ALL_ANNOTATION_METRICS[metric_name](y_true, y_pred, k=0)

    @pytest.mark.parametrize("metric_name", list(ALL_ANNOTATION_METRICS.keys()))
    def test_wrong_input_4(self, metric_name):
        """Missing k."""

        y_true = np.zeros((10, 2, 3))
        y_pred = np.zeros((10, 2, 3))

        with pytest.raises(ValueError):
            ALL_ANNOTATION_METRICS[metric_name](y_true, y_pred, k=1)

    @pytest.mark.parametrize("metric_name", list(ALL_ANNOTATION_METRICS.keys()))
    def test_perfect_prediction(self, metric_name):
        """Perfect prediction.

        Notes
        -----
        The metrics always depends on the class so we need to loop through the classes.
        """

        shape = (2, 8, 9)
        n = shape[0]
        n_classes = 3
        classes = list(range(n_classes))

        y_true = np.empty(shape)
        while set(np.unique(y_true)) != set(
            classes
        ):  # make sure all 3 classes present !!!
            y_true = np.random.randint(n_classes, size=shape)  # 0, 1 and 2 classes

        y_pred = y_true

        for k in classes + [None]:
            metric_average, metric_per_sample = ALL_ANNOTATION_METRICS[metric_name](
                y_true, y_pred, k=k
            )

            if metric_name in ["dice_score", "iou_score"]:
                assert np.all(np.ones(n) == metric_per_sample)
                assert np.nanmean(metric_per_sample) == metric_average

            else:
                raise ValueError("Unrecognized metric name {}".format(metric_name))

    @pytest.mark.parametrize("metric_name", list(ALL_ANNOTATION_METRICS.keys()))
    def test_perfectly_wrong_prediction(self, metric_name):
        """Perfectly wrong prediction."""

        shape = (2, 8, 9)
        n = shape[0]
        n_classes = 3
        classes = list(range(n_classes))

        y_true = np.empty(shape)
        while set(np.unique(y_true)) != set(
            classes
        ):  # make sure all 3 classes present !!!
            y_true = np.random.randint(n_classes, size=shape)  # 0, 1 and 2 classes

        y_pred = np.ones(shape) * n_classes  # perfectly wrong

        for k in classes + [None]:  # None represents label weight
            metric_average, metric_per_sample = ALL_ANNOTATION_METRICS[metric_name](
                y_true, y_pred, k=k
            )

            if metric_name in ["dice_score", "iou_score"]:
                assert np.all(np.zeros(n) == metric_per_sample)
                assert np.nanmean(metric_per_sample) == 0
            else:
                raise ValueError("Unrecognized metric name {}".format(metric_name))


class TestOpticalFlowMetrics:
    """Tests focused on standard optical flow metrics"""

    @pytest.mark.parametrize("c", [1, 4, 0.2])
    def test_angular_norm_independence(self, c):
        """Test that only angle matters for `angular_error_of`."""

        y_true = np.random.random((5, 10, 11, 2))
        y_pred = y_true * c

        metric_average, metric_per_output = angular_error_of(y_true, y_pred)

        assert metric_average == pytest.approx(0, abs=1e-5)
        assert np.allclose(metric_per_output, np.zeros((10, 11)), atol=1e-5)

    @pytest.mark.parametrize("c", [-1, -3, -0.2])
    def test_angular_norm_opposite(self, c):
        """Test that only angle matters for `angular_error_of`."""

        y_true = np.random.random((5, 10, 11, 2))
        y_pred = y_true * c

        metric_average, metric_per_output = angular_error_of(y_true, y_pred)

        assert metric_average == pytest.approx(180, abs=1e-5)
        assert np.allclose(metric_per_output, 180 * np.ones((10, 11)), atol=1e-5)

    @pytest.mark.parametrize("c", [-1, -3, -0.2, 1.2, 8])
    def test_angular_norm_perpendicular(self, c):
        """Test that only angle matters for `angular_error_of`."""

        y_true = np.random.random((5, 10, 11, 2))
        y_pred = np.empty_like(y_true)
        y_pred[..., 0] = y_true[..., 1] * c
        y_pred[..., 1] = -y_true[..., 0] * c

        metric_average, metric_per_output = angular_error_of(y_true, y_pred)

        assert metric_average == pytest.approx(90, abs=1e-5)
        assert np.allclose(metric_per_output, 90 * np.ones((10, 11)), atol=1e-5)

    def test_angular_zero(self):
        """Test that zero vectors lead to nans."""
        y_true = np.zeros((5, 10, 11, 2))
        y_pred = np.ones_like(y_true)

        assert np.all(np.isnan(angular_error_of(y_true, y_pred)[1]))
        assert np.all(np.isnan(angular_error_of(y_pred, y_true)[1]))

    def test_angular_weighted_illegal(self):
        """Test that only possible to use weighing when single sample."""
        y_true = np.zeros((2, 10, 11, 2))
        y_pred = y_true

        with pytest.raises(ValueError):
            angular_error_of(y_true, y_pred, weighted=True)

    def test_angular_weighted_vs_unweighted(self):
        # One 90 (weight 2) and 180 (weight 1) degrees from the x axis
        # nonweighted_average = 135, weighted = 120
        delta_x_true = np.array([[0, 0], [0, -1]])
        delta_y_true = np.array([[2, 0], [0, 0]])

        delta_x_pred = np.ones_like(delta_x_true)  # x axis is the vectors
        delta_y_pred = np.zeros_like(delta_y_true)

        df_true = DisplacementField(delta_x_true, delta_y_true)
        df_pred = DisplacementField(delta_x_pred, delta_y_pred)

        ae_nonweighted, _ = angular_error_of([df_true], [df_pred], weighted=False)
        ae_weighted, _ = angular_error_of([df_true], [df_pred], weighted=True)

        assert ae_nonweighted == pytest.approx(135)
        assert ae_weighted == pytest.approx(120)


class TestIOUScore:
    """Additionally to the TestAnnotationMetrics we write specific tests for the iou score."""

    def test_exclusion_works(self):
        """Test that possible to exclude a class in the segmentation image during iou averaging."""

        random_state = 24
        shape = (10, 15)
        eps = 1e-6

        k = None  # will trigger Iou averaging over all available labels in the true image

        np.random.seed(random_state)

        y_true = np.random.randint(3, size=shape)[np.newaxis, ...]
        y_pred = np.random.randint(3, size=shape)[np.newaxis, ...]

        excluded_labels = [0, 2]

        iou_score_true = (
            np.logical_and(y_true == 1, y_pred == 1).sum()
            / np.logical_or(y_true == 1, y_pred == 1).sum()
        )

        assert (
            abs(
                iou_score_true
                - iou_score(y_true, y_pred, k=k, excluded_labels=excluded_labels)[0]
            )
            < eps
        )


class TestImageSimilarityMetrics:
    """A collection of tests focused on the image similarity metrics.

    Notes
    -----
    Tested against float32 grayscale images.

    """

    @pytest.mark.parametrize(
        "metric",
        [
            pytest.param(x, marks=pytest.mark.slow) if "perceptual_loss" in x else x
            for x in ALL_IMG_METRICS.keys()
        ],
    )
    def test_float_values(self, img_grayscale_float, metric):
        """Test that every image similarity metrics output a float."""

        size = img_grayscale_float.shape
        img_true = img_grayscale_float
        df = DisplacementField.generate(size, approach="affine_simple", rotation=1)
        img_pred = df.warp(img_grayscale_float)

        metric_callable = ALL_IMG_METRICS[metric]

        # UNMASKED
        assert isinstance(metric_callable(img_true, img_pred), float)

        # MASKED
        mask = np.zeros(size, dtype=bool)
        mask[
            size[0] // 3 : 2 * (size[1] // 3), size[0] // 3 : 2 * (size[1] // 3)
        ] = True

        try:
            assert isinstance(metric_callable(img_true, img_pred, mask=mask), float)

        except TypeError:
            # mask keyword argument not present
            pass

    @pytest.mark.parametrize(
        "metric",
        [
            pytest.param(x, marks=pytest.mark.slow) if "perceptual_loss" in x else x
            for x in ALL_IMG_METRICS.keys()
            if x not in ["mi_m", "mi_h"]
        ],
    )
    def test_value_identical_images(self, img_grayscale_float, metric):
        """Test that a specific value is reached if images identical."""

        metric_callable = ALL_IMG_METRICS[metric]

        per_metric_value = {
            "mae": 0,
            "mse": 0,
            "psnr": np.inf,
            "ssmi": 1,
            "mi_m": None,
            "mi_h": None,
            "cross_correlation": 1,
            "perceptual_loss_net_vgg": 0,
            "perceptual_loss_net_alex": 0,
            "perceptual_loss_netlin_vgg": 0,
            "perceptual_loss_netlin_alex": 0,
        }

        assert (
            metric_callable(img_grayscale_float, img_grayscale_float)
            == per_metric_value[metric]
        )

    @pytest.mark.parametrize(
        "metric",
        [
            pytest.param(x, marks=pytest.mark.slow) if "perceptual_loss" in x else x
            for x in ALL_IMG_METRICS.keys()
        ],
    )
    def test_value_different_images(self, img_grayscale_float, metric):
        """Test that for different images losses get bigger and similarities get smaller vs identical images."""

        size = img_grayscale_float.shape
        img_true = img_grayscale_float
        df = DisplacementField.generate(size, approach="affine_simple", rotation=1)
        img_pred = df.warp(img_grayscale_float)

        metric_callable = ALL_IMG_METRICS[metric]

        per_metric_type = {
            "mae": "l",
            "mse": "l",
            "psnr": "s",
            "ssmi": "s",
            "mi_m": "s",
            "mi_h": "s",
            "cross_correlation": "s",
            "perceptual_loss_net_vgg": "l",
            "perceptual_loss_net_alex": "l",
            "perceptual_loss_netlin_vgg": "l",
            "perceptual_loss_netlin_alex": "l",
        }

        if per_metric_type[metric] == "l":
            assert metric_callable(img_true, img_pred) > metric_callable(
                img_true, img_true
            )

        elif per_metric_type[metric] == "s":
            assert metric_callable(img_true, img_pred) < metric_callable(
                img_true, img_true
            )

        else:
            raise ValueError("Invalid metric type")

    @pytest.mark.parametrize(
        "metric",
        [
            pytest.param(x, marks=pytest.mark.slow) if "perceptual_loss" in x else x
            for x in ALL_IMG_METRICS.keys()
        ],
    )
    def test_multisample(self, metric):
        """Test that also possible to pass 3D arrays."""

        n_samples = 2

        img_true = np.random.random((n_samples, 40, 40)).astype("float32")
        img_pred = np.random.random((n_samples, 40, 40)).astype("float32")

        assert (ALL_IMG_METRICS[metric])(img_true, img_pred).shape == (n_samples,)

    @pytest.mark.parametrize(
        "metric",
        [
            pytest.param(x, marks=pytest.mark.slow) if "perceptual_loss" in x else x
            for x in ALL_IMG_METRICS.keys()
            if x not in ["mi_m", "mi_h"]
        ],
    )
    def test_symmetric(self, img_grayscale_float, metric):
        """Test that the order of the input images does not change the metric value.

        Notes
        -----
        Excluding 'mi_m', 'mi_h' because there is some crazy irreproducibility in ANTsPy.

        """
        size = img_grayscale_float.shape
        img_true = img_grayscale_float
        df = DisplacementField.generate(size, approach="affine_simple", rotation=1)
        img_pred = df.warp(img_grayscale_float)

        assert ALL_IMG_METRICS[metric](img_true, img_pred) == ALL_IMG_METRICS[metric](
            img_pred, img_true
        )

    @pytest.mark.parametrize(
        "metric", ["mse", "mae", "psnr", "cross_correlation", "mi_h", "mi_m"]
    )
    def test_mask(self, img_grayscale_float, metric):
        """Test that the mask is working."""
        size = img_grayscale_float.shape
        img_true = img_grayscale_float
        df = DisplacementField.generate(size, approach="affine_simple", rotation=1)

        img_pred_1 = df.warp(img_grayscale_float)

        mask = np.zeros(size, dtype=bool)
        mask[
            size[0] // 3 : 2 * (size[1] // 3), size[0] // 3 : 2 * (size[1] // 3)
        ] = True

        img_pred_2 = img_pred_1.copy()
        img_pred_2[~mask] = np.random.rand()

        eps = 1e-2  # just because of ANTsPy

        assert (
            abs(
                ALL_IMG_METRICS[metric](img_true, img_pred_1, mask=mask)
                - ALL_IMG_METRICS[metric](img_true, img_pred_2, mask=mask)
            )
            < eps
        )

    @pytest.mark.parametrize("metric", list(ALL_IMG_METRICS.keys()))
    def test_incorrect_dtype(self, img_grayscale_float, metric):
        """Test that the type of inputs images has to be float32. If not, raises an exception"""

        img_grayscale_float64 = img_grayscale_float.astype(np.float64)

        with pytest.raises(TypeError):
            ALL_IMG_METRICS[metric](img_grayscale_float, img_grayscale_float64)

    @pytest.mark.parametrize("metric", list(ALL_IMG_METRICS.keys()))
    def test_inconsistent_shape(self, metric):
        """Test that the dimensions of the inputs are consistent. If not, raises an exception."""

        img_true = np.zeros((20, 20, 20), dtype="float32")
        img_pred = np.zeros((10, 20, 20), dtype="float32")
        img_pred2 = np.zeros((20, 10, 50), dtype="float32")

        img_true_2 = np.zeros((20, 20), dtype="float32")
        img_pred_2 = np.zeros((30, 20), dtype="float32")

        with pytest.raises(ValueError):
            ALL_IMG_METRICS[metric](img_true, img_pred)

        with pytest.raises(ValueError):
            ALL_IMG_METRICS[metric](img_true, img_pred2)

        with pytest.raises(ValueError):
            ALL_IMG_METRICS[metric](img_true_2, img_pred_2)

    @pytest.mark.parametrize("metric", list(ALL_IMG_METRICS.keys()))
    def test_inconsistent_mask(self, metric):
        """Test that the dimensions of the mask are consistent. If not, raises an exception"""

        img_true = np.zeros((20, 20, 20), dtype="float32")
        img_pred = np.zeros((10, 20, 20), dtype="float32")
        mask = np.zeros((10, 50), dtype="bool")

        with pytest.raises(ValueError):
            ALL_IMG_METRICS[metric](img_true, img_pred, mask=mask)


class TestKeypointMetrics:
    """Collection of tests focused on the keypoint metrics."""

    def test_incorrect_shape_1(self):
        y_true = np.ones((2, 2))
        y_pred = np.ones((3, 2))

        with pytest.raises(ValueError):
            _euclidean_distance_kp(y_true, y_pred)

        with pytest.raises(ValueError):
            tre_kp(y_true, y_pred)

        with pytest.raises(ValueError):
            rtre_kp(y_true, y_pred, 1, 1)

        with pytest.raises(ValueError):
            improvement_kp(y_true, y_pred, y_pred)

    @pytest.mark.parametrize("random_state", [0, 1, 2])
    def test_weights_sum_up_to_1(self, random_state):
        """Make sure weights sum up to 1 and are nonnegative."""
        np.random.seed(random_state)
        y = np.random.random((10, 2))

        weights = _compute_weights_kp(y)

        assert len(weights) == len(y)
        assert np.all(weights >= 0)
        assert np.sum(weights) == pytest.approx(1)

    @pytest.mark.parametrize("weighted", [True, False])
    def test_dummy_tre(self, weighted):
        """Construct a simple example."""
        y_true = np.array([[1, 1], [0, 0]])
        y_pred = np.array([[0, 1], [1, 0]])

        expected_mean_tre = 1
        expected_tre = np.array([1, 1])
        expected_weights = np.array([0.5, 0.5])

        mean_tre, tre, weights = tre_kp(y_true, y_pred)

        assert expected_mean_tre == mean_tre
        assert np.allclose(expected_tre, tre)
        assert np.allclose(expected_weights, weights)

    def test_unit_diagonal(self):
        """Make sure rtre = tre if the diagonal is sized 1."""
        n_keypoints = 4
        y_pred = np.random.random((n_keypoints, 2))
        y_true = np.random.random((n_keypoints, 2))

        h = w = (1 / 2) ** (1 / 2)

        mean_tre, tre, weights_tre = tre_kp(y_true, y_pred)
        mean_rtre, rtre, weights_rtre = rtre_kp(y_true, y_pred, h, w)

        assert len(tre) == len(rtre) == n_keypoints
        assert mean_tre == mean_rtre
        assert np.allclose(tre, rtre)
        assert np.allclose(weights_tre, weights_rtre)

    def test_improvement(self):
        """Test perfect improvement and perfect not improvement."""
        y_true = np.array([[i, i] for i in range(4)])

        assert improvement_kp(y_true, y_true + 2, y_true + 3)[0] == 1
        assert improvement_kp(y_true, y_true + 3, y_true + 1)[0] == 0
