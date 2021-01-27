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

from atlalign.utils import _find_all_children, find_labels_dic


class TestSegmentationConcatenation:
    """A set of methods testing the concatenation of labels for the segmentation."""

    def test_find_children(self, label_dict):
        dic = {"id": 2, "children": []}

        assert np.all(_find_all_children(dic) == [2])
        assert np.all(set(_find_all_children(label_dict)) == {2, 3, 4, 5, 6})

    @pytest.mark.parametrize("depth", [0, 1, 10])
    def test_background(self, depth, label_dict):
        shape = (20, 20)
        segmentation_array = np.zeros(shape)
        new_segmentation_array = np.zeros(shape)

        assert np.all(
            find_labels_dic(segmentation_array, label_dict, depth)
            == new_segmentation_array
        )

    @pytest.mark.parametrize("depth", [0, 1, 10])
    def test_labels_not_in_tree(self, depth, label_dict):
        shape = (20, 20)
        segmentation_array = np.ones(shape) * 10
        new_segmentation_array = np.ones(shape) * -1

        assert np.all(
            find_labels_dic(segmentation_array, label_dict, depth)
            == new_segmentation_array
        )

    @pytest.mark.parametrize("label", [2, 3, 4, 5, 6])
    def test_tree_concatenation(self, label, label_dict):
        depth = 0
        shape = (20, 20)
        segmentation_array = np.ones(shape) * label
        new_segmentation_array = np.ones(shape) * 2

        assert np.all(
            find_labels_dic(segmentation_array, label_dict, depth)
            == new_segmentation_array
        )

    @pytest.mark.parametrize("unchanged_label", [2, 3, 4])
    def test_tree_concatenation_unchanged_labels(self, unchanged_label, label_dict):
        depth = 1
        shape = (20, 20)
        segmentation_array = np.ones(shape) * unchanged_label
        new_segmentation_array = np.ones(shape) * unchanged_label

        assert np.all(
            find_labels_dic(segmentation_array, label_dict, depth)
            == new_segmentation_array
        )

    def test_specific_example(self, label_dict):
        """A simple 3 x 3 matrix segmentation array."""

        segmentation_array = np.array([[[1, 2, 3], [20, 14, 50], [4, 5, 6]]])

        segmentation_array_d0 = np.array([[[-1, 2, 2], [-1, -1, -1], [2, 2, 2]]])

        segmentation_array_d1 = np.array([[[-1, 2, 3], [-1, -1, -1], [4, 4, 4]]])

        segmentation_array_d2 = np.array([[[-1, 2, 3], [-1, -1, -1], [4, 5, 6]]])

        segmentation_array_d3 = np.array([[[-1, 2, 3], [-1, -1, -1], [4, 5, 6]]])

        assert np.all(
            find_labels_dic(segmentation_array, label_dict, 0) == segmentation_array_d0
        )
        assert np.all(
            find_labels_dic(segmentation_array, label_dict, 1) == segmentation_array_d1
        )
        assert np.all(
            find_labels_dic(segmentation_array, label_dict, 2) == segmentation_array_d2
        )
        assert np.all(
            find_labels_dic(segmentation_array, label_dict, 3) == segmentation_array_d3
        )
