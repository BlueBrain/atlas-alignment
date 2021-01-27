"""Collection of tools for aggregating slices to 3D models."""

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
import scipy

from atlalign.data import nissl_volume


class Volume:
    """Class representing mutliple slices.

    Parameters
    ----------
    sn : list
        List of section numbers.

    mov_imgs : list
        List of np.ndarrays representing the moving images corresponding to the `sn`.

    dvfs : list
        List of displacement fields corresponding to the `sn`.
    """

    def __init__(self, sn, mov_imgs, dvfs):
        # initial checks
        if not len(sn) == len(mov_imgs) == len(dvfs):
            raise ValueError("All the input lists need to have the same length")

        if len(set(sn)) != len(sn):
            raise ValueError("There are duplicate section numbers.")

        if not all([0 <= x < 528 for x in sn]):
            raise ValueError("All section numbers must lie in [0, 528).")

        self.sn = sn
        self.mov_imgs = mov_imgs
        self.dvfs = dvfs

        # attributes
        self.ref_imgs = nissl_volume()[self.sn, ..., 0]
        self.sn_to_ix = [
            None if x not in self.sn else self.sn.index(x) for x in range(528)
        ]
        self.reg_imgs = self._warp()

    @property
    def sorted_dvfs(self):
        """Return displacement fields sorted by the coronal section."""
        sorted_sn = sorted(self.sn)

        return [self.dvfs[self.sn_to_ix[s]] for s in sorted_sn], sorted_sn

    @property
    def sorted_mov(self):
        """Return moving images as sorted by the coronal section."""
        sorted_sn = sorted(self.sn)

        return [self.mov_imgs[self.sn_to_ix[s]] for s in sorted_sn], sorted_sn

    @property
    def sorted_ref(self):
        """Return reference images as sorted by the coronal section."""
        sorted_sn = sorted(self.sn)

        return [self.ref_imgs[self.sn_to_ix[s]] for s in sorted_sn], sorted_sn

    @property
    def sorted_reg(self):
        """Return registered images as sorted by the coronal section."""
        sorted_sn = sorted(self.sn)

        return [self.reg_imgs[self.sn_to_ix[s]] for s in sorted_sn], sorted_sn

    def _warp(self):
        """Warp the moving images to get registered ones."""
        return [df.warp(img) for df, img in zip(self.dvfs, self.mov_imgs)]

    def __getitem__(self, key):
        """Get all relevant data for a specified section.

        Parameters
        ----------
        key : int
            Section number to query.

        Returns
        -------
        ref_img : np.ndarray
            Reference image.

        mov_img : np.ndarray
            Moving image.

        reg_img : np.ndarray
            Registered image.

        df : DisplacementField
            Displacement field (mov2reg).
        """
        if self.sn_to_ix[key] is None:
            raise KeyError("The section {} not found".format(key))

        ix = self.sn_to_ix[key]

        return self.ref_imgs[ix], self.mov_imgs[ix], self.reg_imgs[ix], self.dvfs[ix]


class GappedVolume:
    """Volume containing gaps.

    Parameters
    ----------
    sn : list
        List of section numbers. Note that not required to be ordered.

    imgs : np.ndarray or list
        Internally converted to list of grayscale images of same shape representing different coronal sections.
        Order corresponds to the one in `sn`.

    """

    def __init__(self, sn, imgs):

        if isinstance(imgs, np.ndarray):
            # turn into a list
            imgs = np.squeeze(imgs)
            imgs = [imgs[i] for i in range(len(imgs))]

        # checks
        if len(sn) != len(imgs):
            raise ValueError("Inconsitent lenghts")

        if len({img.shape for img in imgs}) != 1:
            raise ValueError("All the images need to have the same shape")

        if len(sn) != len(set(sn)):
            raise ValueError("There are duplicates in section numbers.")

        self.sn = sn
        self.imgs = imgs

        self.shape = imgs[0].shape


class CoronalInterpolator:
    """Interpolator that works pixel by pixel in the coronal dimension."""

    def __init__(self, kind="linear", fill_value=0, bounds_error=False):
        """Construct."""
        self.kind = kind
        self.fill_value = fill_value
        self.bounds_error = bounds_error

    def interpolate(self, gv):
        """Interpolate.

        Note that some section images might have pixels equal to np.nan. In this case these pixels are skipped in the
        interpolation.

        Parameters
        ----------
        gv : GappedVolume
            Instance of the ``GappedVolume`` to be interpolated.

        Returns
        -------
        final_volume : np.ndarray
            Array of shape (528, 320, 456) that holds the entire interpolated volume without gaps.

        """
        grid = np.array(range(528))
        final_volume = np.empty((len(grid), *gv.shape))

        for r in range(gv.shape[0]):
            for c in range(gv.shape[1]):
                x_pixel, y_pixel = zip(
                    *[
                        (s, img[r, c])
                        for s, img in zip(gv.sn, gv.imgs)
                        if not np.isnan(img[r, c])
                    ]
                )

                f = scipy.interpolate.interp1d(
                    x_pixel,
                    y_pixel,
                    kind=self.kind,
                    bounds_error=self.bounds_error,
                    fill_value=self.fill_value,
                )
                try:
                    final_volume[:, r, c] = f(grid)
                except Exception as e:
                    print(e)

        return final_volume
