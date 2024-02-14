"""Collection of tests focused on the `volume` module."""

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

from unittest.mock import MagicMock

import numpy as np
import pytest

from atlalign.base import DisplacementField
from atlalign.volume import CoronalInterpolator, GappedVolume, Volume


@pytest.fixture()
def minimal_vol(monkeypatch):
    sn = [1, 13]
    mov_imgs = 2 * [np.zeros((12, 13))]
    dvfs = 2 * [DisplacementField.generate((12, 13), approach="identity")]

    nvol_mock = MagicMock()
    nvol_mock.__getitem__.return_value = np.zeros((len(mov_imgs), 12, 13, 1))
    monkeypatch.setattr("atlalign.volume.nissl_volume", lambda: nvol_mock)

    return Volume(sn, mov_imgs, dvfs)


class TestVolume:
    def test_wrong_input(self):
        with pytest.raises(ValueError):
            Volume([None], [], [])

        with pytest.raises(ValueError):
            Volume(["a", "a"], ["a", "a"], ["a", "a"])

        with pytest.raises(ValueError):
            Volume([1111, 12], ["a", "a"], ["a", "a"])

    def test_construction(self, minimal_vol):
        assert isinstance(minimal_vol, Volume)

    def test_getitem(self, minimal_vol):
        with pytest.raises(KeyError):
            minimal_vol[4]

        outputs = minimal_vol[1]

        assert len(outputs) == 4
        assert isinstance(outputs[0], np.ndarray)
        assert isinstance(outputs[1], np.ndarray)
        assert isinstance(outputs[2], np.ndarray)
        assert isinstance(outputs[3], DisplacementField)

    def test_sorted_attributes(self, minimal_vol):
        assert isinstance(minimal_vol.sorted_dvfs[0], list)
        assert isinstance(minimal_vol.sorted_mov[0], list)
        assert isinstance(minimal_vol.sorted_reg[0], list)
        assert isinstance(minimal_vol.sorted_ref[0], list)


class TestGappedVolume:
    def test_incorrect_input(self):
        with pytest.raises(ValueError):
            GappedVolume([1], [])

        with pytest.raises(ValueError):
            GappedVolume([1, 1], [np.zeros((1, 2)), np.zeros((2, 3))])

    def test_array2list_conversion(self):
        sn = [1, 44, 12]
        shape = (10, 11)
        imgs = np.array([np.zeros(shape) for _ in range(len(sn))])

        gv = GappedVolume(sn, imgs)

        assert isinstance(gv.sn, list)
        assert np.allclose(sn, gv.sn)
        assert isinstance(gv.imgs, list)


class TestCoronalInterpolator:
    @pytest.mark.parametrize(
        "kind", ["linear", "nearest", "zero", "slinear", "previous", "next"]
    )
    def test_all_kinds(self, kind):
        ip = CoronalInterpolator(kind=kind, fill_value=0, bounds_error=False)

        sn = [0, 527]
        imgs = np.zeros((len(sn), 10, 11))
        dummy_gv = GappedVolume(sn, imgs)

        final_volume = ip.interpolate(dummy_gv)

        assert np.allclose(np.zeros((528, *dummy_gv.shape)), final_volume)

    @pytest.mark.parametrize(
        "kind",
        [
            "linear",
            "quadratic",
            "cubic",
            "nearest",
            "zero",
            "slinear",
            "previous",
            "next",
        ],
    )
    def test_precise_on_known(self, kind):
        """Make sure that on the known slices the interpolation is precise."""
        ip = CoronalInterpolator(kind=kind, fill_value=0, bounds_error=False)

        shape = (10, 11)
        sn = list(range(0, 528, 8)) + [527]
        imgs = np.random.random((len(sn), *shape))

        gv = GappedVolume(sn, imgs)

        final_volume = ip.interpolate(gv)

        for i, s in enumerate(sn):
            assert np.allclose(imgs[i], final_volume[s])

    @pytest.mark.parametrize(
        "kind", ["linear", "nearest", "zero", "slinear", "previous", "next"]
    )
    def test_nan_all(self, kind):
        """Make sure that if one input section composed fully of NaN pixels then things work."""
        ip = CoronalInterpolator(kind=kind, fill_value=0, bounds_error=False)

        shape = (10, 11)
        sn = [0, 100, 527]
        imgs = [np.zeros(shape), np.ones(shape) * np.nan, np.zeros(shape)]
        gv = GappedVolume(sn, imgs)

        final_volume = ip.interpolate(gv)

        assert np.all(np.isfinite(final_volume))

    @pytest.mark.parametrize(
        "kind", ["linear", "nearest", "zero", "slinear", "previous", "next"]
    )
    def test_nan_some(self, kind):
        """Make sure that if input section has a NaN pixel then things work."""
        ip = CoronalInterpolator(kind=kind, fill_value=0, bounds_error=False)

        shape = (10, 11)
        sn = [0, 100, 527]

        valid = np.ones(shape, dtype=bool)
        valid[5:9, 2:7] = False

        weird_img = np.random.random(shape)
        weird_img[5:9, 2:7] = np.nan
        imgs = [np.zeros(shape), weird_img, np.zeros(shape)]

        gv = GappedVolume(sn, imgs)

        final_volume = ip.interpolate(gv)

        assert np.all(np.isfinite(final_volume))
        assert np.allclose(final_volume[sn[1]][valid], weird_img[valid])
