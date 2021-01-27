"""Collection of tests focused on the augmentations.py module."""

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
from unittest.mock import Mock

import numpy as np
import pytest

from atlalign.augmentations import DatasetAugmenter, load_dataset_in_memory
from atlalign.base import DisplacementField


@pytest.mark.parametrize(
    "key",
    [
        "dataset_id",
        "deltas_xy",
        "image_id",
        "img",
        "inv_deltas_xy",
        "p",
    ],
)
def test_load_dataset_in_memory(path_test_data, key):
    h5_path = path_test_data / "supervised_dataset.h5"

    res = load_dataset_in_memory(h5_path, key)

    assert isinstance(res, np.ndarray)
    assert len(res) > 0


class TestDatasetAugmenter:
    def test_construction(self, path_test_data):
        da = DatasetAugmenter(path_test_data / "supervised_dataset.h5")

        assert da.n_orig > 0

    @pytest.mark.parametrize("n_iter", [1, 2])
    @pytest.mark.parametrize("anchor", [True, False])
    @pytest.mark.parametrize("is_valid", [True, False])
    def test_augment(
        self, monkeypatch, path_test_data, tmpdir, n_iter, anchor, is_valid
    ):
        fake_es = Mock(
            return_value=DisplacementField.generate(
                (320, 456), approach="affine_simple", rotation=0.2
            )
        )
        max_corrupted_pixels = 10

        if not is_valid:
            max_corrupted_pixels = 0  # hack that will force to use the original

        monkeypatch.setattr("atlalign.zoo.edge_stretching", fake_es)

        da = DatasetAugmenter(path_test_data / "supervised_dataset.h5")

        output_path = pathlib.Path(str(tmpdir)) / "output.h5"

        da.augment(
            output_path,
            n_iter=n_iter,
            anchor=anchor,
            max_trials=2,
            max_corrupted_pixels=max_corrupted_pixels,
            ds_f=32,
        )

        assert output_path.exists()

        keys = ["dataset_id", "deltas_xy", "image_id", "img", "inv_deltas_xy", "p"]
        for key in keys:
            array = load_dataset_in_memory(output_path, key)
            assert da.n_orig * n_iter == len(array)
            assert not np.any(np.isnan(array))

        keys = ["dataset_id", "image_id", "p"]
        for key in keys:
            original_a = load_dataset_in_memory(da.original_path, key)
            new_a = load_dataset_in_memory(output_path, key)
            new_a_expected = np.concatenate(n_iter * [original_a])

            np.testing.assert_equal(new_a, new_a_expected)
