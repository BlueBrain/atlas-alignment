"""Module creating one-to-many augmentations."""

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

import h5py
import numpy as np
from skimage.feature import canny
from skimage.util import img_as_float32

from atlalign.base import DisplacementField


def load_dataset_in_memory(h5_path, dataset_name):
    """Load a dataset of a h5 file in memory."""
    with h5py.File(h5_path, "r") as f:
        return f[dataset_name][:]


class DatasetAugmenter:
    """Class that does the augmentation.

    Attributes
    ----------
    original_path : str
        Path to where the original dataset is located.

    """

    def __init__(self, original_path):
        self.original_path = original_path

        self.n_orig = len(load_dataset_in_memory(self.original_path, "image_id"))

    def augment(
        self,
        output_path,
        n_iter=10,
        anchor=True,
        p_reg=0.5,
        random_state=None,
        max_corrupted_pixels=500,
        ds_f=8,
        max_trials=5,
    ):
        """Augment the original dataset and create a new one.

        Note that this not modify the original dataset.

        Parameters
        ----------
        output_path : str
            Path to where the new h5 file stored.

        n_iter : int
            Number of augmented samples per each sample in the original dataset.

        anchor : bool
            If True, then dvf anchored before inverted.

        p_reg : bool
            Probability that we start from a registered image
            (rather than the moving).

        random_state : bool
            Random state

        max_corrupted_pixels : int
            Maximum numbr of corrupted pixels allowed for a dvf - the actual
            number is computed as np.sum(df.jacobian() < 0)

        ds_f : int
            Downsampling factor for inverses. 1 creates the least artifacts.

        max_trials : int
            Max number of attemps to augment before an identity displacement
            used as augmentation.
        """
        np.random.seed(random_state)

        n_new = n_iter * self.n_orig
        print(n_new)

        with h5py.File(self.original_path, "r") as f_orig:
            # extract
            dset_img_orig = f_orig["img"]
            dset_image_id_orig = f_orig["image_id"]
            dset_dataset_id_orig = f_orig["dataset_id"]
            dset_deltas_xy_orig = f_orig["deltas_xy"]
            dset_inv_deltas_xy_orig = f_orig["inv_deltas_xy"]
            dset_p_orig = f_orig["p"]

            with h5py.File(output_path, "w") as f_aug:
                dset_img_aug = f_aug.create_dataset(
                    "img", (n_new, 320, 456), dtype="uint8"
                )
                dset_image_id_aug = f_aug.create_dataset(
                    "image_id", (n_new,), dtype="int"
                )
                dset_dataset_id_aug = f_aug.create_dataset(
                    "dataset_id", (n_new,), dtype="int"
                )
                dset_p_aug = f_aug.create_dataset("p", (n_new,), dtype="int")
                dset_deltas_xy_aug = f_aug.create_dataset(
                    "deltas_xy", (n_new, 320, 456, 2), dtype=np.float16
                )
                dset_inv_deltas_xy_aug = f_aug.create_dataset(
                    "inv_deltas_xy", (n_new, 320, 456, 2), dtype=np.float16
                )

                for i in range(n_new):
                    print(i)
                    i_orig = i % self.n_orig

                    mov2reg = DisplacementField(
                        dset_deltas_xy_orig[i_orig, ..., 0],
                        dset_deltas_xy_orig[i_orig, ..., 1],
                    )

                    # copy
                    dset_image_id_aug[i] = dset_image_id_orig[i_orig]
                    dset_dataset_id_aug[i] = dset_dataset_id_orig[i_orig]
                    dset_p_aug[i] = dset_p_orig[i_orig]

                    use_reg = np.random.random() > p_reg
                    print("Using registered: {}".format(use_reg))

                    if not use_reg:
                        # mov != reg
                        img_mov = dset_img_orig[i_orig]
                    else:
                        # mov=reg
                        img_mov = mov2reg.warp(dset_img_orig[i_orig])
                        mov2reg = DisplacementField.generate(
                            (320, 456), approach="identity"
                        )

                    is_nice = False
                    n_trials = 0

                    while not is_nice:
                        n_trials += 1

                        if n_trials == max_trials:
                            print("Replicating original: out of trials")
                            dset_img_aug[i] = dset_img_orig[i_orig]
                            dset_deltas_xy_aug[i] = dset_deltas_xy_orig[i_orig]
                            dset_inv_deltas_xy_aug[i] = dset_inv_deltas_xy_orig[i_orig]
                            break

                        else:
                            mov2art = self.generate_mov2art(img_mov)

                        reg2mov = mov2reg.pseudo_inverse(ds_f=ds_f)
                        reg2art = reg2mov(mov2art)

                        # anchor
                        if anchor:
                            print("ANCHORING")
                            reg2art = reg2art.anchor(
                                ds_f=50, smooth=0, h_kept=0.9, w_kept=0.9
                            )

                        art2reg = reg2art.pseudo_inverse(ds_f=ds_f)

                        validity_check = np.all(
                            np.isfinite(reg2art.delta_x)
                        ) and np.all(np.isfinite(reg2art.delta_y))
                        validity_check &= np.all(
                            np.isfinite(art2reg.delta_x)
                        ) and np.all(np.isfinite(art2reg.delta_y))
                        jacobian_check = (
                            np.sum(reg2art.jacobian < 0) < max_corrupted_pixels
                        )
                        jacobian_check &= (
                            np.sum(art2reg.jacobian < 0) < max_corrupted_pixels
                        )

                        if validity_check and jacobian_check:
                            is_nice = True
                            print("Check passed")
                        else:
                            print("Check failed")

                    if n_trials != max_trials:
                        dset_img_aug[i] = mov2art.warp(img_mov)
                        dset_deltas_xy_aug[i] = np.stack(
                            [art2reg.delta_x, art2reg.delta_y], axis=-1
                        )
                        dset_inv_deltas_xy_aug[i] = np.stack(
                            [reg2art.delta_x, reg2art.delta_y], axis=-1
                        )

    @staticmethod
    def generate_mov2art(img_mov, verbose=True, radius_max=60, use_normal=True):
        """Generate geometric augmentation and its inverse."""
        shape = img_mov.shape
        img_mov_float = img_as_float32(img_mov)
        edge_mask = canny(img_mov_float)

        if use_normal:
            c = np.random.normal(0.7, 0.3)
        else:
            c = np.random.random()

        if verbose:
            print("Scalar: {}".format(c))

        mov2art = c * DisplacementField.generate(
            shape,
            approach="edge_stretching",
            edge_mask=edge_mask,
            interpolation_method="rbf",
            interpolator_kwargs={"function": "linear"},
            n_perturbation_points=6,
            radius_max=radius_max,
        )

        return mov2art
