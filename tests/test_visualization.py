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
from matplotlib import animation

from atlalign.base import DisplacementField
from atlalign.visualization import (
    create_animation,
    create_segmentation_image,
    generate_df_plots,
)


class TestUtils:
    """A collection of tests focues on the `utils` module."""

    def test_wrong_input_dtype(self):
        """Make sure that float input dtype not allowed."""

        shape = (4, 5)

        segmentation_array = np.ones(shape, dtype=np.float32) / 5

        with pytest.raises(TypeError):
            create_segmentation_image(segmentation_array)

    def test_correct_output_type(self):
        """Make sure that uint8 output dtype."""

        shape = (4, 5)

        segmentation_array = np.random.randint(100, size=shape)

        segmentation_img, _ = create_segmentation_image(segmentation_array)

        assert segmentation_img.dtype == np.uint8
        assert np.all((0 <= segmentation_img) & (segmentation_img < 256))

    def test_different_classes_different_colors(self):
        """Test different classes have different colors."""

        segmentation_array = np.array([[0, 10], [2, 0]])

        segmentation_img, _ = create_segmentation_image(segmentation_array)

        assert (
            len(
                {
                    tuple(x)
                    for x in [
                        segmentation_img[0, 0],
                        segmentation_img[0, 1],
                        segmentation_img[1, 0],
                        segmentation_img[1, 1],
                    ]
                }
            )
            == 3
        )

        assert np.all(segmentation_img[0, 0] == segmentation_img[1, 1])

    def test_predefined_colors(self):
        """Test possible to pass colors."""

        segmentation_array = np.array([[0, 1], [2, 22]])

        colors_dict = {0: (0, 0, 0), 1: (255, 0, 0)}

        segmentation_img, _ = create_segmentation_image(segmentation_array, colors_dict)

        assert np.all(segmentation_img[0, 0] == (0, 0, 0))
        assert np.all(segmentation_img[0, 1] == (255, 0, 0))

    def test_animation(self, img):
        """Possible to generate animations."""

        df = DisplacementField.generate(img.shape, approach="identity")

        ani = create_animation(df, img)
        ani_many = create_animation([df, df], img)

        assert isinstance(ani, animation.Animation)
        assert isinstance(ani_many, animation.Animation)


class TestGenerateDFPlots:
    """Tests focused on the `generate_df_plots` function."""

    @pytest.mark.parametrize("df_id", [(320, 456)], indirect=True)
    def test_basic(self, df_id, tmpdir, monkeypatch):
        generate_df_plots(df_id, df_id, tmpdir)
