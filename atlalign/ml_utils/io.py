"""Collection of functions dealing with input and output."""

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

import copy
import datetime
import pathlib

import h5py
import keras
import mlflow
import numpy as np

from atlalign.base import DisplacementField
from atlalign.data import nissl_volume


class SupervisedGenerator(keras.utils.Sequence):
    """Generator streaming supervised data from a HDF5 files.

    Parameters
    ----------
    hdf_path : str or pathlib.Path
        Path to where the hdf5 file is.

    batch_size : int
        Batch size.

    shuffle : bool
        If True, then data shuffled at the end of each epoch and also in the constructor.

    augmenter_ref : None or imgaug.augmenters.Sequential
        If None, no augmentation. If instance of a imgaug `Sequential` then its a pipeline that will be used
        to augment all reference images in a batch.

    augmenter_mov : None or imgaug.augmenters.Sequential
        If None, no augmentation. If instance of a imgaug `Sequential` then its a pipeline that will be used
        to augment all moving images in a batch.

    return_inverse : bool
        If True, then targets are [img_reg, dvf, dvf_inv] else its only [img_reg, dvf].

    indexes : None or list or str/pathlib.Path
        A user defined list of indices to be used for streaming. This is used to give user the chance
        to only stream parts of the data. If str or pathlib.Path then read from a .npy file

    Attributes
    ----------
    indexes : list
        List of indices determining the slicing order.

    volume : np.array
        Array representing the nissle stain volume of dtype float32.

    times : list
        List of timedeltas representing the for each batch yeild.
    """

    def __init__(
        self,
        hdf_path,
        batch_size=32,
        shuffle=False,
        augmenter_ref=None,
        augmenter_mov=None,
        return_inverse=False,
        mlflow_log=False,
        indexes=None,
    ):

        self._locals = locals()
        del self._locals["self"]
        self._locals["augmenter_mov"] = None if augmenter_mov is None else "active"
        self._locals["augmenter_ref"] = None if augmenter_ref is None else "active"

        if mlflow_log:
            mlflow.log_params(self._locals)

        self.hdf_path = str(hdf_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenter_ref = augmenter_ref
        self.augmenter_mov = augmenter_mov
        self.return_inverse = return_inverse

        with h5py.File(self.hdf_path, "r") as f:
            length = len(f["p"])

        if indexes is None:
            self.indexes = list(np.arange(length))

        elif isinstance(indexes, list):
            self.indexes = indexes

        elif isinstance(indexes, (str, pathlib.Path)):
            self.indexes = list(np.load(str(indexes)))

        else:
            raise TypeError("Invalid indexes type {}".format(type(indexes)))

        self.volume = nissl_volume()

        self.times = []
        self.temp = []

        self.on_epoch_end()

    def __len__(self):
        """Length of the iterator = number of steps per epoch."""
        n_samples = len(self.indexes)

        return int(np.floor(n_samples / self.batch_size))

    def __getitem__(self, index):
        """Load samples in memory and possibly augment.

        Parameters
        ----------
        index : int
            Integer representing the index of to be returned batch.

        Returns
        -------
        X : np.array
            Array of shape (`self.batch_size`, 320, 456, 2) representing the stacked samples of reference
            and moving images.

        targets : list
            If `self.return_inverse=False` then 2 element list. First element represents the true registered images
            - shape (`self.batch_size`, 320, 456, 1). The second element is a batch of ground truth displacement
            vector fields  - shape (`self.batch_size`, 320, 456, 2).
            If `self.return_inverse=True` then 3 element list. The first two elements are like above and the
            third one is a batch of ground truth inverse displacements fields (warping images in reference space to
            moving space) of shape (`self.batch_size`, 320, 456, 2).
        """
        begin_time = datetime.datetime.now()
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        sorted_indexes = sorted(indexes)  # hdf5 only supports sorted indexing

        # Generate indexes of the batch
        with h5py.File(self.hdf_path, "r") as f:
            dset_img = f["img"]
            dset_deltas_xy = f["deltas_xy"]
            dset_inv_deltas_xy = f["inv_deltas_xy"]
            dset_p = f["p"]

            sn = np.minimum(
                dset_p[sorted_indexes] // 25, np.ones(len(indexes), dtype=int) * 527
            )
            ref_images = self.volume[sn]
            mov_images = (
                dset_img[sorted_indexes][..., np.newaxis].astype("float32") / 255
            )
            batch_deltas_xy = dset_deltas_xy[sorted_indexes]
            batch_inv_deltas_xy = (
                dset_inv_deltas_xy[sorted_indexes] if self.return_inverse else None
            )

        if self.augmenter_ref is not None:
            ref_images = self.augmenter_ref.augment_images(ref_images)

        if self.augmenter_mov is not None:
            mov_images = self.augmenter_mov.augment_images(mov_images)

        X = np.concatenate([ref_images, mov_images], axis=3)
        if self.return_inverse:
            X_mr = np.concatenate([mov_images, ref_images], axis=3)

        # Registered images
        reg_images = np.zeros_like(mov_images)

        for i in range(len(mov_images)):
            df = DisplacementField(
                batch_deltas_xy[i, ..., 0], batch_deltas_xy[i, ..., 1]
            )
            assert df.is_valid, "{} is not valid".format(sorted_indexes[i])
            reg_images[i, ..., 0] = df.warp(mov_images[i, ..., 0])

        self.times.append((datetime.datetime.now() - begin_time))

        if self.return_inverse:

            return [X, X_mr], [reg_images, batch_deltas_xy, batch_inv_deltas_xy]
        else:
            return X, [reg_images, batch_deltas_xy]

    def on_epoch_end(self):
        """Take end of epoch action."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_all_data(self):
        """Load entire dataset into memory."""
        orig_params = copy.deepcopy(self._locals)

        orig_params["batch_size"] = 1
        orig_params["shuffle"] = False
        orig_params["mlflow_log"] = False

        new_gen = self.__class__(**orig_params)

        all_inps, all_outs = [], []
        for inps, outs in new_gen:
            all_inps.append(inps)
            all_outs.append(outs)

        return all_inps, all_outs
