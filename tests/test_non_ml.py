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
from atlalign.non_ml import antspy_registration


class TestAntspyRegistration:
    """A collection of tests focused on the ANTsPy registration."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "registration_type",
        [
            "Translation",
            "Rigid",
            "Similarity",
            "QuickRigid",
            "DenseRigid",
            "BOLDRigid",
            "Affine",
            "AffineFast",
            "BOLDAffine",
            "TRSAA",
            "ElasticSyN",
            "SyN",
            "SyNRA",
            "SyNOnly",
            "SyNCC",
            "SyNabp",
            "SyNBold",
            "SyNBoldAff",
            "SyNAggro",
            "TVMSQ",
        ],
    )
    def test_each_type(self, tmp_path, img_grayscale, registration_type):
        """Make sure that every type of registration gives a valid DisplacementField as output."""

        moving_img = img_grayscale
        fixed_img = img_grayscale

        reg_iterations = (4, 2, 0)

        df_final, meta = antspy_registration(
            fixed_img,
            moving_img,
            registration_type=registration_type,
            reg_iterations=reg_iterations,
            path=tmp_path,
        )

        assert isinstance(df_final, DisplacementField)
        assert df_final.is_valid

    def test_displacement_field(self, tmp_path, img_grayscale):
        """Make sure that the displacement extracted can reproduce the registered moving image.

        Done by comparing with the image contained in the registration output.
        """

        fixed_img = img_grayscale
        size = fixed_img.shape
        df = DisplacementField.generate(
            size, approach="affine_simple", translation_x=20, translation_y=20
        )
        moving_img = df.warp(img_grayscale)

        df_final, meta = antspy_registration(fixed_img, moving_img, path=tmp_path)
        img1 = meta["warpedmovout"].numpy()
        img2 = df_final.warp(
            moving_img, interpolation="linear", border_mode="constant", c=0
        )

        if img_grayscale.dtype == "uint8":
            epsilon = 1
        else:
            epsilon = 0.005

        assert abs(img1 - img2).mean() < epsilon

    @pytest.mark.todo
    @pytest.mark.parametrize(
        "registration_type",
        [
            "Translation",
            "Rigid",
            "Similarity",
            "QuickRigid",
            "DenseRigid",
            "BOLDRigid",
            "Affine",
            "AffineFast",
            "BOLDAffine",
            "TRSAA",
            "ElasticSyN",
            "SyN",
            "SyNRA",
            "SyNOnly",
            "SyNCC",
            "SyNabp",
        ],
    )
    def test_same_results(
        self, tmp_path, monkeypatch, img_grayscale, registration_type
    ):
        """Make sure that the registration is reproducible.

        Done by checking if the displacement field extracted are equal if we run the registration with
        the same parameters.

        Notes
        -----
        Marked as `todo` because we did not find a way how to force ANTsPY to be always reproducible.
        """

        monkeypatch.setenv("ANTS_RANDOM_SEED", "1")
        monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

        size = img_grayscale.shape
        p = img_grayscale / np.sum(img_grayscale)
        df = DisplacementField.generate(size, approach="paper", p=p, random_state=1)
        moving_img = df.warp(img_grayscale)

        df_final1, meta1 = antspy_registration(
            img_grayscale,
            moving_img,
            registration_type=registration_type,
            path=tmp_path,
            verbose=True,
        )
        df_final2, meta2 = antspy_registration(
            img_grayscale,
            moving_img,
            registration_type=registration_type,
            path=tmp_path,
            verbose=True,
        )

        assert np.allclose(df_final1.delta_x, df_final2.delta_x, atol=1)
        assert np.allclose(df_final1.delta_y, df_final2.delta_y, atol=1)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "registration_type",
        [
            "Translation",
            "Rigid",
            "Similarity",
            "QuickRigid",
            "DenseRigid",
            "BOLDRigid",
            "Affine",
            "AffineFast",
            "BOLDAffine",
            "TRSAA",
            "ElasticSyN",
            "SyN",
            "SyNRA",
            "SyNOnly",
            "SyNCC",
            "SyNabp",
            "SyNBold",
            "SyNBoldAff",
            "SyNAggro",
            "TVMSQ",
        ],
    )
    def test_different_results(self, tmp_path, img_grayscale_uint, registration_type):
        """Make sure that the registration is not reproducible if some environment variables not set.


        The environment variables are ANTS_RANDOM_SEED and ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS.
        """

        size = img_grayscale_uint.shape
        df = DisplacementField.generate(
            size, approach="affine_simple", translation_x=20
        )
        moving_img = df.warp(img_grayscale_uint)

        df_final1, meta1 = antspy_registration(
            img_grayscale_uint, moving_img, path=tmp_path
        )
        df_final2, meta2 = antspy_registration(
            img_grayscale_uint, moving_img, path=tmp_path
        )

        assert not np.allclose(df_final1.delta_x, df_final2.delta_x, atol=0.1)
        assert not np.allclose(df_final1.delta_y, df_final2.delta_y, atol=0.1)

    @pytest.mark.todo
    def test_different_types(
        self, tmp_path, monkeypatch, img_grayscale_uint, img_grayscale_float
    ):
        """Make sure that the registration does not depend on the type of input images.

        Notes
        -----
        Marked as `todo` because we did not find a way how to force ANTsPY to be always reproducible.
        """

        monkeypatch.setenv("ANTS_RANDOM_SEED", "4")
        monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

        size = img_grayscale_uint.shape
        p = img_grayscale_uint / np.sum(img_grayscale_uint)
        df = DisplacementField.generate(size, approach="paper", p=p)
        moving_img_uint = df.warp(img_grayscale_uint)
        moving_img_float = df.warp(img_grayscale_float)

        df_final1, meta1 = antspy_registration(
            img_grayscale_uint, moving_img_uint, path=tmp_path, verbose=False
        )
        df_final2, meta2 = antspy_registration(
            img_grayscale_float, moving_img_float, path=tmp_path, verbose=False
        )

        assert np.allclose(df_final1.delta_x, df_final2.delta_x, atol=0.1)
        assert np.allclose(df_final1.delta_y, df_final2.delta_y, atol=0.1)
