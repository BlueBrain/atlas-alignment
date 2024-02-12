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

import pathlib

import numpy as np
import pytest

from atlalign.data import (
    annotation_volume,
    circles,
    manual_registration,
    nissl_volume,
    rectangles,
    segmentation_collapsing_labels,
)
from atlalign.utils import _find_all_children


class TestAnnotationVolume:
    """A collection of tests focused on the `annotation_volume` function."""

    def test_load_works(self, monkeypatch, tmpdir):
        """Test that loading works"""
        # Lets do some patching
        monkeypatch.setattr(
            "numpy.load",
            lambda *args, **kwargs: np.zeros((528, 320, 456), dtype=np.float32),
        )

        x_atlas = annotation_volume()

        # Final output
        assert x_atlas.shape == (528, 320, 456)
        assert np.all(np.isfinite(x_atlas))
        assert x_atlas.dtype == np.int32


class TestNisslVolume:
    """A collection of tests focused on the `nissl_volume` function."""

    def test_load_works(self, monkeypatch):
        """Test that loading works."""
        # Lets do some patching
        monkeypatch.setattr(
            "numpy.load",
            lambda *args, **kwargs: np.zeros((528, 320, 456), dtype=np.float32),
        )
        monkeypatch.setattr(
            "atlalign.data.img_as_float32",
            lambda *args, **kwargs: np.zeros((320, 456), dtype=np.float32),
        )

        x_atlas = nissl_volume()

        # Final output
        assert x_atlas.shape == (528, 320, 456, 1)
        assert np.all(np.isfinite(x_atlas))
        assert x_atlas.min() >= 0
        assert x_atlas.max() <= 1
        assert x_atlas.dtype == np.float32


class TestManualRegistration:
    def test_correct_keys(self, monkeypatch):
        path = "tests/data/supervised_dataset.h5"  # manual patch

        res = manual_registration(path)

        # Final output
        assert set(res.keys()) == {
            "dataset_id",
            "deltas_xy",
            "image_id",
            "img",
            "inv_deltas_xy",
            "p",
        }
        assert np.all([isinstance(x, np.ndarray) for x in res.values()])


class TestCircles:
    """Collection of tests focues on the circles function."""

    def test_input_shape(self):
        """Test that only allows for 2D."""

        shape_wrong = (41, 21, 312)
        shape_correct = (100, 30)

        with pytest.raises(ValueError):
            circles(10, shape_wrong, 10)

        circles(10, shape_correct, 10)

    def test_correct_dtype(self):
        """Test that float32 ndarray is returned."""

        shape = (100, 30)
        res = circles(10, shape, 10)

        assert res.min() >= 0
        assert res.max() <= 1
        assert res.dtype == np.float32

    @pytest.mark.parametrize("n_levels", [1, 2, 3, 4, 5, 6])
    def test_correct_number_of_intensities(self, n_levels):
        """Test whether n_unique_intensities = n_levels + 1"""

        res = circles(10, (200, 220), (40, 50), n_levels=n_levels)

        for row in res:
            assert len(np.unique(row) == n_levels + 1)  # also count black background

    def test_reproducible(self):
        """Test that random_state works."""

        res_1 = circles(10, (100, 120), (20, 40), n_levels=(2, 6), random_state=None)
        res_2 = circles(10, (100, 120), (20, 40), n_levels=(2, 6), random_state=1)
        res_3 = circles(10, (100, 120), (20, 40), n_levels=(2, 6), random_state=2)
        res_4 = circles(10, (100, 120), (20, 40), n_levels=(2, 6), random_state=1)
        res_5 = circles(10, (100, 120), (20, 40), n_levels=(2, 6), random_state=None)

        assert np.all(res_2 == res_4)
        assert not np.all(res_2 == res_3)
        assert not np.all(res_2 == res_1)
        assert not np.all(res_3 == res_1)
        assert not np.all(res_1 == res_5)


class TestRectangles:
    """A collection of tests testing the rectangles function."""

    def test_input_shape(self):
        """Test that only allows for 2D."""

        shape_wrong = (40, 30, 10)
        shape_correct = (40, 30)

        with pytest.raises(ValueError):
            rectangles(100, shape_wrong, 10, 20)

        rectangles(100, shape_correct, 10, 20)

    def test_wrong_type(self):
        """Test that height, width and n_levels only work with integers."""

        shape = (40, 50)
        with pytest.raises(TypeError):
            rectangles(100, shape, 10.1, 20)

        with pytest.raises(TypeError):
            rectangles(100, shape, 10, 20.3)

        with pytest.raises(TypeError):
            rectangles(100, shape, (10, 20), 20, n_levels=3.4)

        rectangles(10, shape, 10, 20, n_levels=2)

    def test_wrong_rectangle_size(self):
        """Test that rectangle needs to fit the image."""

        shape = (img_h, img_w) = (40, 50)

        with pytest.raises(ValueError):
            rectangles(100, shape, img_h + 1, img_w - 1)

        with pytest.raises(ValueError):
            rectangles(100, shape, img_h - 1, img_w + 1)

        with pytest.raises(ValueError):
            rectangles(100, shape, img_h + 1, img_w + 1)

        rectangles(100, shape, img_h - 1, img_w - 1)

    def test_wrong_n_levels(self):
        """Test that n_levels needs to be correct."""

        with pytest.raises(ValueError):
            rectangles(5, (100, 120), 20, 10, n_levels=14)

        with pytest.raises(ValueError):
            rectangles(5, (100, 120), 10, 20, n_levels=14)

        rectangles(5, (100, 120), 20, 20, n_levels=14)

    def test_reproducible(self):
        """Test that random_state works."""

        res_1 = rectangles(
            10, (100, 120), (20, 30), (10, 40), n_levels=(1, 4), random_state=None
        )
        res_2 = rectangles(
            10, (100, 120), (20, 30), (10, 40), n_levels=(1, 4), random_state=1
        )
        res_3 = rectangles(
            10, (100, 120), (20, 30), (10, 40), n_levels=(1, 4), random_state=2
        )
        res_4 = rectangles(
            10, (100, 120), (20, 30), (10, 40), n_levels=(1, 4), random_state=1
        )
        res_5 = rectangles(
            10, (100, 120), (20, 30), (10, 40), n_levels=(1, 4), random_state=None
        )

        assert np.all(res_2 == res_4)
        assert not np.all(res_2 == res_3)
        assert not np.all(res_2 == res_1)
        assert not np.all(res_3 == res_1)
        assert not np.all(res_1 == res_5)

    @pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4])
    def test_no_empty_images(self, random_state):
        """Test that no empty images."""

        shape = (100, 120)
        res = rectangles(
            10, shape, (20, 30), (10, 50), n_levels=4, random_state=random_state
        )

        zeros = np.zeros((*shape, 1))
        for row in res:
            assert not np.all(row == zeros)

    def test_output_shape(self):

        """Test that the shape of the output is correct."""
        shape = (50, 100)
        res = rectangles(10, shape, (20, 30), (10, 50), n_levels=4)

        assert res.shape == (10, *shape, 1)

    def test_correct_dtype(self):
        """Test that float32 ndarray is returned."""

        shape = (50, 100)
        res = rectangles(10, shape, (20, 30), (10, 50), n_levels=4)

        assert res.min() >= 0
        assert res.max() <= 1
        assert res.dtype == np.float32

    def test_full_intensity(self):
        """Test that there exists a pixel with a full intensity = 1."""

        shape = (50, 100)
        res = rectangles(10, shape, (20, 30), (10, 50), n_levels=4)

        for row in res:
            assert np.any(row == 1)

    @pytest.mark.parametrize("n_levels", [1, 2, 3, 4, 5, 6])
    def test_correct_number_of_intensities(self, n_levels):
        """Test whether n_unique_intensities = n_levels + 1"""

        res = rectangles(10, (200, 220), (40, 50), (50, 100), n_levels=n_levels)

        for row in res:
            assert len(np.unique(row) == n_levels + 1)  # also count black background


class TestSegmentationCollapsingLabels:
    def test_load_works(self, monkeypatch, tmpdir):
        """Test that loading works."""
        tmpfile = pathlib.Path(str(tmpdir)) / "temp.json"
        tmpfile.touch()

        # Lets do some patching
        monkeypatch.setattr("json.load", lambda *args, **kwargs: {})

        res = segmentation_collapsing_labels(tmpfile)

        # Final output
        assert isinstance(res, dict)

    @pytest.mark.skip
    def test_no_id_equal_to_negative_one(self):
        """Make sure that -1 is not an existing label since we want to use it as a default not found value.

        We skip this bacause it assumes the dataset is downloaded and it wouldnt make sense to patch this.

        See Also
        --------
        """

        all_children = _find_all_children(segmentation_collapsing_labels())

        assert isinstance(all_children, list)
        assert all_children
        assert -1 not in all_children
