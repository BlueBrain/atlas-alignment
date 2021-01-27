"""Tests focused on callbacks."""

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
from unittest.mock import Mock

import h5py
import mlflow
import numpy as np
import pandas as pd
import pytest
from keras.models import Model

from atlalign.augmentations import load_dataset_in_memory
from atlalign.ml_utils import (
    MLFlowCallback,
    SupervisedGenerator,
    get_mlflow_artifact_path,
)
from atlalign.nn import supervised_model_factory


def test_get_mlflow_artifact_path(monkeypatch, tmpdir):
    monkeypatch.chdir(tmpdir)

    with mlflow.start_run():
        artifact_path = get_mlflow_artifact_path()

    expected_artifact_path = pathlib.Path(str(tmpdir))
    expected = [
        x for x in pathlib.Path(str(tmpdir)).rglob("*") if "artifacts" in str(x)
    ][0]

    assert artifact_path == expected


class TestMLFlowCallback:
    @staticmethod
    def create_h5(h5_path, n_samples, random_state):
        height, width = 320, 456
        np.random.seed(random_state)

        with h5py.File(h5_path, "w") as f:
            dset_img = f.create_dataset(
                "img", (n_samples, height, width), dtype="uint8"
            )
            dset_image_id = f.create_dataset("image_id", (n_samples,), dtype="int")
            dset_dataset_id = f.create_dataset("dataset_id", (n_samples,), dtype="int")
            dset_p = f.create_dataset("p", (n_samples,), dtype="int")
            dset_deltas_xy = f.create_dataset(
                "deltas_xy",
                (n_samples, height, width, 2),
                dtype=np.float16,
                fillvalue=0,
            )
            dset_inv_deltas_xy = f.create_dataset(
                "inv_deltas_xy",
                (n_samples, height, width, 2),
                dtype=np.float16,
                fillvalue=0,
            )

            # Populate
            dset_deltas_xy[:] = np.random.random((n_samples, height, width, 2))
            dset_inv_deltas_xy[:] = np.random.random((n_samples, height, width, 2))
            dset_img[:] = np.random.randint(
                0, high=255, size=(n_samples, height, width)
            )
            dset_image_id[:] = 50 + np.random.choice(
                n_samples, size=n_samples, replace=False
            )
            dset_dataset_id[:] = 1000 + np.random.choice(
                n_samples, size=n_samples, replace=False
            )
            dset_p[:] = np.random.randint(0, high=12000, size=n_samples)

    def test_hooks(self, tmpdir, path_test_data, monkeypatch):
        tmpdir = pathlib.Path(str(tmpdir))
        monkeypatch.chdir(tmpdir)
        fake_sg_c = Mock(return_value=Mock(spec=SupervisedGenerator))
        monkeypatch.setattr(
            "atlalign.ml_utils.callbacks.SupervisedGenerator", fake_sg_c
        )
        monkeypatch.setattr("atlalign.ml_utils.callbacks.keras", Mock())

        h5_path = path_test_data / "supervised_dataset.h5"

        train_ixs_path = "a"
        val_ixs_path = "b"

        with mlflow.start_run():
            cb = MLFlowCallback(h5_path, train_ixs_path, val_ixs_path, freq=2)

            assert fake_sg_c.call_count == 2

            # Train
            train_kwargs = fake_sg_c.call_args_list[0][1]
            assert train_kwargs["indexes"] == train_ixs_path
            assert train_kwargs["shuffle"] == False
            assert train_kwargs["batch_size"] == 1

            # Val
            val_kwargs = fake_sg_c.call_args_list[1][1]
            assert val_kwargs["indexes"] == val_ixs_path
            assert val_kwargs["shuffle"] == False
            assert val_kwargs["batch_size"] == 1

            # On train begin
            assert not (cb.root_path / "architecture").exists()
            assert not (cb.root_path / "checkpoints").exists()

            cb.on_train_begin()

            assert (cb.root_path / "architecture").exists()
            assert (cb.root_path / "checkpoints").exists()

            # On batch_end
            cb.model = Mock(metrics_names=[])  # Inject a keras model
            cb.model.evaluate_generator.return_value = []
            monkeypatch.setattr(
                cb,
                "compute_external_metrics",
                Mock(return_value=pd.Series({2: "a"}).to_frame()),
            )

            cb.on_batch_end(None)  # 1
            cb.on_batch_end(None)  # 2

    @pytest.mark.parametrize("random_state", [3, 10])
    @pytest.mark.parametrize("return_inverse", [True, False])
    def test_compute_external_metrics(
        self, monkeypatch, tmpdir, random_state, return_inverse
    ):
        evaluate_cache = []

        def fake_evaluate(*args, **kwargs):
            evaluate_cache.append(
                {
                    "deltas_true": args[0],
                    "img_mov": args[2],
                    "p": kwargs["p"],
                    "deltas_true_inv": kwargs["deltas_true_inv"],
                }
            )

            return pd.Series([2, 3])

        monkeypatch.setattr(
            "atlalign.ml_utils.callbacks.evaluate_single",
            Mock(side_effect=fake_evaluate),
        )
        monkeypatch.setattr("atlalign.ml_utils.callbacks.annotation_volume", Mock())
        monkeypatch.setattr(
            "atlalign.ml_utils.io.nissl_volume",
            Mock(return_value=np.zeros((528, 320, 456, 1))),
        )
        monkeypatch.setattr(
            "atlalign.ml_utils.callbacks.segmentation_collapsing_labels", Mock()
        )

        n_samples = 10
        n_val_samples = 4
        h5_path = pathlib.Path(str(tmpdir)) / "temp.h5"
        self.create_h5(h5_path, n_samples, random_state)

        val_indexes = list(np.random.choice(n_samples, n_val_samples, replace=False))

        val_gen = SupervisedGenerator(
            h5_path,
            indexes=val_indexes,
            shuffle=False,
            batch_size=1,
            return_inverse=return_inverse,
        )
        losses = ["mse", "mse", "mse"] if return_inverse else ["mse", "mse"]
        losses_weights = [1, 1, 1] if return_inverse else [1, 1]

        model = supervised_model_factory(
            compute_inv=return_inverse,
            losses=losses,
            losses_weights=losses_weights,
            start_filters=(2,),
            downsample_filters=(4, 2),
            middle_filters=(2,),
            upsample_filters=(2, 4),
        )

        df = MLFlowCallback.compute_external_metrics(model, val_gen)

        assert len(df) == len(val_indexes)
        assert np.allclose(
            df.index.values, load_dataset_in_memory(h5_path, "image_id")[val_indexes]
        )
        assert len(evaluate_cache) == len(val_indexes)

        for ecache, val_index in zip(evaluate_cache, val_indexes):
            expected_deltas = load_dataset_in_memory(h5_path, "deltas_xy")[val_index]
            expected_deltas_inv = load_dataset_in_memory(h5_path, "inv_deltas_xy")[
                val_index
            ]
            expected_image = load_dataset_in_memory(h5_path, "img")[val_index] / 255
            expected_p = load_dataset_in_memory(h5_path, "p")[val_index]

            assert np.allclose(expected_deltas, ecache["deltas_true"])
            assert np.allclose(expected_image, ecache["img_mov"])
            assert np.allclose(expected_p, ecache["p"])

            if return_inverse:
                assert np.allclose(expected_deltas_inv, ecache["deltas_true_inv"])
            else:
                assert ecache["deltas_true_inv"] is None  # they are not streamed
