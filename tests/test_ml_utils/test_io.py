"""Tests focues on the atlalign.io module."""

"""
    The package atlalign is a tool for registration of 2D images.

    Copyright (C) 2021 EPFL/Blue Brain Project

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pathlib

import imgaug.augmenters as iaa
import numpy as np
import pytest

from atlalign.ml_utils import SupervisedGenerator

SUPERVISED_H5_PATH = (
    pathlib.Path(__file__).parent.parent / "data" / "supervised_dataset.h5"
)


@pytest.fixture(scope="session")
def fake_nissl_volume():
    return np.random.random((528, 320, 456, 1)).astype(np.float32)


class TestSupervisedGenerator:
    def test_inexistent_h5(self, monkeypatch, fake_nissl_volume):
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            lambda *args, **kwargs: fake_nissl_volume,
        )
        path_wrong = SUPERVISED_H5_PATH.parent / "fake.h5"

        with pytest.raises(OSError):
            SupervisedGenerator(path_wrong)

    def test_correct_indexes(self, monkeypatch, fake_nissl_volume):
        """Inner indices are [0, 1]."""
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            lambda *args, **kwargs: fake_nissl_volume,
        )

        assert SupervisedGenerator(SUPERVISED_H5_PATH).indexes == [0, 1]

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_length(self, batch_size, monkeypatch, fake_nissl_volume):
        """Test that length of the Sequence is computed correctly."""
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            lambda *args, **kwargs: fake_nissl_volume,
        )
        correct_len = 2 // batch_size

        assert (
            len(SupervisedGenerator(SUPERVISED_H5_PATH, batch_size=batch_size))
            == correct_len
        )

    def test_shuffling(self, monkeypatch, fake_nissl_volume):
        """Test shuffling works."""
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            lambda *args, **kwargs: fake_nissl_volume,
        )
        n_trials = 10
        is_different = False
        gen = SupervisedGenerator(SUPERVISED_H5_PATH, shuffle=True)
        orig_indexes = gen.indexes[:]

        for _ in range(n_trials):
            gen.on_epoch_end()
            is_different = is_different or orig_indexes != gen.indexes

        assert is_different

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_getitem(self, batch_size, monkeypatch, fake_nissl_volume):
        """Get item."""
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            lambda *args, **kwargs: fake_nissl_volume,
        )
        gen = SupervisedGenerator(SUPERVISED_H5_PATH, batch_size=batch_size)

        inp, out = gen[0]

        assert inp.shape == (batch_size, 320, 456, 2)
        assert inp.dtype == np.float32
        assert 0 <= inp.min() <= inp.max() <= 1

        assert isinstance(out, list)
        assert len(out) == 2

        assert out[0].shape == (batch_size, 320, 456, 1) and out[1].shape == (
            batch_size,
            320,
            456,
            2,
        )
        assert out[0].dtype == np.float32 and out[1].dtype == np.float16
        assert 0 <= out[0].min() <= out[0].max() <= 1

    @pytest.mark.parametrize("aug_ref", [True, False])
    @pytest.mark.parametrize("aug_mov", [True, False])
    def test_augmenters(self, aug_ref, aug_mov, monkeypatch, fake_nissl_volume):
        """Augmenting works."""
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            lambda *args, **kwargs: fake_nissl_volume,
        )
        batch_size = 2
        augmenter = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontal flips
                iaa.Crop(percent=(0, 0.1)),  # random crops
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            ]
        )

        kwargs = {
            "augmenter_ref": augmenter if aug_ref else None,
            "augmenter_mov": augmenter if aug_mov else None,
        }

        gen = SupervisedGenerator(SUPERVISED_H5_PATH, batch_size=batch_size, **kwargs)

        inp, out = gen[0]

        assert inp.dtype == np.float32

        assert isinstance(out, list)
        assert len(out) == 2

        assert out[0].shape == (batch_size, 320, 456, 1) and out[1].shape == (
            batch_size,
            320,
            456,
            2,
        )
        assert out[0].dtype == np.float32 and out[1].dtype == np.float16

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_get_all_data(self, batch_size, monkeypatch, fake_nissl_volume):
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            lambda *args, **kwargs: fake_nissl_volume,
        )

        gen = SupervisedGenerator(SUPERVISED_H5_PATH, batch_size=batch_size)

        all_inp, all_out = gen.get_all_data()

        assert len(all_inp) == 2
        assert len(all_out) == 2

    def test_indexes(self, monkeypatch, fake_nissl_volume):
        """Make sure indexes attribute work."""
        batch_size = 1
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            lambda *args, **kwargs: fake_nissl_volume,
        )

        gen_indexes = SupervisedGenerator(
            SUPERVISED_H5_PATH, batch_size=batch_size, indexes=[0]
        )
        gen = SupervisedGenerator(SUPERVISED_H5_PATH, batch_size=batch_size)

        assert len(gen) == 2
        assert len(gen_indexes) == 1
