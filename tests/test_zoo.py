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

from atlalign.base import DisplacementField


class TestAffine:
    """A collection of tests focused on the affine transformations."""

    @pytest.mark.parametrize("direction", ["U", "D", "L", "R", "DR", "UL"])
    def test_translation(self, direction):
        """Make sure that one can translate.

        Notes
        -----
        As well as in our context, in scikit-image positive delta_y means moving down.
        """

        shape = (40, 50)
        if direction == "U":
            matrix = np.array([[1, 0, 0], [0, 1, -1]])
            delta_x = np.zeros(shape)
            delta_y = -np.ones(shape)

        elif direction == "D":
            matrix = np.array([[1, 0, 0], [0, 1, 1]])
            delta_x = np.zeros(shape)
            delta_y = np.ones(shape)

        elif direction == "L":
            matrix = np.array([[1, 0, -1], [0, 1, 0]])
            delta_x = -np.ones(shape)
            delta_y = np.zeros(shape)

        elif direction == "R":
            matrix = np.array([[1, 0, 1], [0, 1, 0]])
            delta_x = np.ones(shape)
            delta_y = np.zeros(shape)

        elif direction == "DR":
            matrix = np.array([[1, 0, 1], [0, 1, 1]])
            delta_x = np.ones(shape)
            delta_y = np.ones(shape)

        elif direction == "UL":
            matrix = np.array([[1, 0, -3], [0, 1, -3]])
            delta_x = -np.ones(shape) * 3
            delta_y = -np.ones(shape) * 3

        else:
            raise ValueError("Unsupported direction {}".format(direction))

        df = DisplacementField.generate(shape, approach="affine", matrix=matrix)

        assert df == DisplacementField(delta_x, delta_y)

    @pytest.mark.parametrize("scale", [1, 1.1, 0.9, 2])
    def test_scale(self, scale):
        """Make sure that one can scale."""

        shape = (40, 50)
        matrix = scale * np.array([[1, 0, 0], [0, 1, 0]])

        x, y = np.meshgrid(list(range(shape[1])), list(range(shape[0])))
        delta_x = scale * x - x
        delta_y = scale * y - y

        df = DisplacementField.generate(shape=shape, approach="affine", matrix=matrix)
        df_true = DisplacementField(delta_x, delta_y)

        assert df == df_true


class TestEdgeStretching:
    """A collection of tests focused on the edge_stretching algorithm"""

    def test_no_edges(self):
        """Test that no edges lead to identity mapping."""

        shape = (20, 13)
        # No edges
        df_1 = DisplacementField.generate(
            shape,
            approach="edge_stretching",
            edge_mask=np.zeros(shape, dtype=bool),
            n_perturbation_points=10,
        )

        # No perturbation points
        df_2 = DisplacementField.generate(
            shape,
            approach="edge_stretching",
            edge_mask=np.ones(shape, dtype=bool),
            n_perturbation_points=0,
        )

        df_id = DisplacementField(np.zeros(shape), np.zeros(shape))

        assert df_id == df_1
        assert df_id == df_2

    def test_illegal_kwarg(self):
        """Test that illegal kwarg not accepted"""

        with pytest.raises(ValueError):
            shape = (20, 13)
            DisplacementField.generate(
                shape,
                approach="edge_stretching",
                edge_mask=np.zeros(shape, dtype=bool),
                n_perturbation_points=10,
                i_am_illegal=1,
            )


class TestControlPoints:
    """A collection of tests focused on the control_points approach."""

    def test_wrong_inputs_0(self):
        """Test that other types then np.ndarray lead to an error"""

        shape = (40, 50)
        points = "fake"
        values_delta_x = (1, 2, 3)
        values_delta_y = [1, 1312, "aaa"]

        with pytest.raises(TypeError):
            DisplacementField.generate(
                shape,
                approach="control_points",
                points=points,
                values_delta_x=values_delta_x,
                values_delta_y=values_delta_y,
            )

    def test_wrong_inputs_1(self):
        """Test that inconsistent input lengths lead to an error."""

        shape = (40, 50)
        points = np.array([[0, 1], [10, 10]])  # len(points) = 2
        values_delta_x = np.array([0, 10, 0])  # len(values_delta_x) = 3
        values_delta_y = np.array([0, 10])  # len(values_delta_y) = 2

        with pytest.raises(ValueError):
            DisplacementField.generate(
                shape,
                approach="control_points",
                points=points,
                values_delta_x=values_delta_x,
                values_delta_y=values_delta_y,
            )

    def test_wrong_inputs_2(self):
        """Test that no control points are not allowed.

        Notes
        -----
        Note that if all 3 inputs are None and anchor_corners is True,
        then automatically generates 4 control points in the background.
        """

        shape = (40, 50)
        points = np.zeros((0, 2))
        values_delta_x = np.array([])
        values_delta_y = np.array([])

        with pytest.raises(ValueError):
            DisplacementField.generate(
                shape,
                approach="control_points",
                points=points,
                values_delta_x=values_delta_x,
                values_delta_y=values_delta_y,
            )

    def test_wrong_inputs_3(self):
        """Test that other dimensions are also correct."""

        shape = (40, 50)
        points = np.zeros([1, 2, 3, 4, 5])  # Wrong - needs to be a 2d array
        values_delta_x = np.array([1, 2, 3, 4, 5])
        values_delta_y = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            DisplacementField.generate(
                shape,
                approach="control_points",
                points=points,
                values_delta_x=values_delta_x,
                values_delta_y=values_delta_y,
            )

    def test_wrong_inputs_4(self):
        """Test that out of range control points lead to an error."""

        shape = (40, 50)
        points = np.array([[100, 10], [12, 12]])  # Wrong - needs to be a 2d array
        values_delta_x = np.array([1, 2])
        values_delta_y = np.array([0, 13])

        with pytest.raises(IndexError):
            DisplacementField.generate(
                shape,
                approach="control_points",
                points=points,
                values_delta_x=values_delta_x,
                values_delta_y=values_delta_y,
            )

    def test_no_inputs_anchor_corners(self):
        """Test that can/cannot instantiate with no inputs and with/without anchoring corners."""

        shape = (40, 50)
        with pytest.raises(ValueError):
            DisplacementField.generate(
                shape, approach="control_points", anchor_corners=False
            )

        DisplacementField.generate(
            shape, approach="control_points", anchor_corners=True
        )

    @pytest.mark.parametrize(
        "interpolation_input",
        [
            ("griddata", {"method": "nearest"}),
            ("griddata", {"method": "linear"}),
            ("griddata", {"method": "cubic"}),
            ("griddata_custom", {}),
            ("bspline", {}),
            ("rbf", {"function": "multiquadric"}),
            ("rbf", {"function": "inverse"}),
            ("rbf", {"function": "gaussian"}),
            ("rbf", {"function": "linear"}),
            ("rbf", {"function": "cubic"}),
            ("rbf", {"function": "quintic"}),
            ("rbf", {"function": "thin-plate"}),
        ],
    )
    def test_control_points_correspond(self, interpolation_input):
        """Test that the interpolation is precise on control points.

        Notes
        -----
        This is a terrible test since we are using SmoothSplines and they do not necessarily construct
        a interpolation such that it is equal to the known values on the control points. So if the equality tests
        are failing (which they probably will for more control points) then it does not mean our scheme is not working.
        """

        interpolation_method, interpolator_kwargs = interpolation_input
        eps = 1e-5  # This is negligible since the control point values are integers (see below)
        shape = (30, 20)
        n_pixels = np.prod(shape)
        n_control_points = 16
        random_state = 12

        # For bspline to work there needs to be at least (16 points)
        np.random.seed(random_state)

        points = np.array(
            [
                [x // shape[1], x % shape[1]]
                for x in np.random.choice(
                    n_pixels, size=n_control_points, replace=False
                )
            ]
        )

        values_delta_x = np.random.randint(10, size=n_control_points)
        values_delta_y = np.random.randint(11, size=n_control_points)

        df = DisplacementField.generate(
            shape,
            approach="control_points",
            points=points,
            values_delta_x=values_delta_x,
            values_delta_y=values_delta_y,
            anchor_corners=False,
            interpolation_method=interpolation_method,
            interpolator_kwargs=interpolator_kwargs,
        )

        for i, p in enumerate(points):
            assert abs(values_delta_x[i] - df.delta_x[p[0], p[1]]) < eps
            assert abs(values_delta_y[i] - df.delta_y[p[0], p[1]]) < eps


@pytest.mark.parametrize(
    "approach",
    [
        "affine",
        "affine_simple",
        "control_points",
        "paper",
        "microsoft",
        "patch_shift",
        "projective",
        "single_frequency",
    ],
)
def test_construction(approach):
    """Just check default factory methods are able to construct the class with no additional parameters."""

    shape = (500, 500)
    inst = DisplacementField.generate(shape, approach=approach)

    assert isinstance(inst, DisplacementField)


@pytest.mark.parametrize("random_state", [0, 1, 2, 3])
@pytest.mark.parametrize("approach", ["paper", "microsoft"])
def test_reproducible(random_state, approach):
    """Test that the same random seeds lead to identical result whereas different ones do not."""
    shape = (100, 120)

    df_1 = DisplacementField.generate(
        shape, approach=approach, random_state=random_state
    )
    df_2 = DisplacementField.generate(
        shape, approach=approach, random_state=random_state
    )
    df_3 = DisplacementField.generate(
        shape, approach=approach, random_state=random_state + 1
    )

    assert df_1 == df_2
    assert df_1 != df_3
