"""Callbacks and aggregation functions."""

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

import h5py
import keras
import mlflow
import pandas as pd

from atlalign.data import annotation_volume, segmentation_collapsing_labels
from atlalign.metrics import evaluate_single
from atlalign.ml_utils.io import SupervisedGenerator


def get_mlflow_artifact_path(start_char=7):
    """Get path to the MLFlow artifacts of the active run.

    Stupid implementation.

    Parameters
    ----------
    start_char : int
        Since the string will start like "file:///actual/path..." we just
        slice it.
    """
    return pathlib.Path(mlflow.active_run().info.artifact_uri[start_char:])


class MLFlowCallback(keras.callbacks.Callback):
    """Logs metrics into ML.

    Notes
    -----
    Only runs inside of an mlflow context.

    Parameters
    ----------
    merged_path : str
        Path to the master h5 file containing all the data.

    train_original_ixs_path : str
        Path to where original training indices are stored.

    val_original_ixs_path : str
        Path to where original validation indices are stored.

    freq : int
        Reports metrics on every `freq` batch.

    workers : int
        Number of workers to be used for each of the evaluations.

    return_inverse : bool
        If True, then generators behave differently.

    starting_step : int
        Useful when we want to use a checkpointed model and log metrics as of a different step then 1.

    use_validation : int
        If True, then the custom metrics are computed on the validation set.
        Otherwise they will be computed on the training set.
    """

    def __init__(
        self,
        merged_path,
        train_original_ixs_path,
        val_original_ixs_path,
        freq=10,
        workers=1,
        return_inverse=False,
        starting_step=0,
        use_validation=True,
    ):
        super().__init__()

        # Check if inside of an mlflow context
        if mlflow.active_run() is None:
            raise ValueError(
                "To use the MLFlowCallback one needs to be inside of a mlflow.start_run context."
            )

        # mlflow
        self.root_path = get_mlflow_artifact_path()
        mlflow.log_params(
            {
                "train_original_ixs_path": train_original_ixs_path,
                "val_original_ixs_path": val_original_ixs_path,
                "merged_path": merged_path,
            }
        )

        self.train_original_gen = SupervisedGenerator(
            merged_path,
            indexes=train_original_ixs_path,
            shuffle=False,
            batch_size=1,
            return_inverse=return_inverse,
        )

        self.val_original_gen = SupervisedGenerator(
            merged_path,
            indexes=val_original_ixs_path,
            shuffle=False,
            batch_size=1,
            return_inverse=return_inverse,
        )
        self.freq = freq

        self.workers = workers
        self.overall_batch = starting_step
        self.use_validation = use_validation

    def on_train_begin(self, logs=None):
        """Save model architecture."""
        arch_path = self.root_path / "architecture"
        checkpoints_path = self.root_path / "checkpoints"

        arch_path.mkdir(parents=True, exist_ok=True)
        checkpoints_path.mkdir(parents=True, exist_ok=True)

    def on_batch_end(self, batch, logs=None):
        """Log metrics to mlflow.

        The goal here is two extract 3 types of metrics:
            - train_merged - extracted from logs (it is a running average over epoch)
            - train_original - computed via evaluate_generator
            - val_original - computed via evaluate_generator
        """
        self.overall_batch += 1

        if self.overall_batch % self.freq != 0:
            return

        model = self.model
        metric_names = model.metrics_names

        all_metrics = {}

        # Keras
        all_metrics.update(
            {"{}_train_merged".format(metric): logs[metric] for metric in metric_names}
        )

        eval_train_original = model.evaluate_generator(
            self.train_original_gen, workers=self.workers
        )
        all_metrics.update(
            {
                "{}_train_original".format(metric): value
                for metric, value in zip(metric_names, eval_train_original)
            }
        )

        eval_val_original = model.evaluate_generator(
            self.val_original_gen, workers=self.workers
        )
        all_metrics.update(
            {
                "{}_val_original".format(metric): value
                for metric, value in zip(metric_names, eval_val_original)
            }
        )

        # Custom
        print(
            "\nComputing custom metrics on {} set!".format(
                "val" if self.use_validation else "train"
            )
        )
        gen = self.val_original_gen if self.use_validation else self.train_original_gen

        external_metrics_df = self.compute_external_metrics(model, gen)

        stats_dir = self.root_path / str(self.overall_batch) / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        external_metrics_df.to_csv(str(stats_dir / "stats.csv"))
        external_metrics_df.to_html(str(stats_dir / "stats.html"))

        external_metrics = dict(external_metrics_df.mean())
        all_metrics.update(external_metrics)

        # log into mlflow
        mlflow.log_metrics(all_metrics, step=self.overall_batch)

        keras.models.save_model(
            model,
            str(
                self.root_path
                / "checkpoints"
                / "model_{}.h5".format(self.overall_batch)
            ),
        )

    @staticmethod
    def compute_external_metrics(model, gen):
        """Compute external matrics sample by sample.

        Parameters
        ----------
        model
            Keras model

        gen : SupervisedGenerator
            Generator

        Returns
        -------
        metrics : dict
            Various metrics.
        """
        # checks
        if gen.shuffle:
            raise ValueError("Shuffling is not allowed for external metrics!")
        if gen.batch_size != 1:
            raise ValueError("Batch size has to be 1 for external metrics")

        # Prepare annotation related stuff (load in RAM, small arrays)
        indexes = gen.indexes
        with h5py.File(gen.hdf_path, "r") as f:
            ps = f["p"][:][indexes]
            ids = f["image_id"][:][indexes]

        avol = annotation_volume()
        collapsing_labels = segmentation_collapsing_labels()

        external_metrics_per_sample = []

        for i, p in enumerate(ps):
            sample = gen[i]  # data[indexes[i]]
            if gen.return_inverse:
                img_mov = sample[0][0][0, ..., 1]
                deltas_true = sample[1][1][0]
                deltas_true_inv = sample[1][2][0]

            else:
                img_mov = sample[0][0, ..., 1]
                deltas_true = sample[1][1][0]
                deltas_true_inv = None

            deltas_pred = model.predict(sample[0])[1][0]
            deltas_pred_inv = None  # we do not use the predicted one

            # compute external metrics
            res, images = evaluate_single(
                deltas_true,
                deltas_pred,
                img_mov,
                ds_f=8,  # orig 8
                p=p,
                deltas_true_inv=deltas_true_inv,
                deltas_pred_inv=deltas_pred_inv,
                avol=avol,
                collapsing_labels=collapsing_labels,
                depths=(0, 1, 2, 3, 4, 5, 6, 7, 8),
            )
            external_metrics_per_sample.append(res)

        external_metrics_df = pd.DataFrame(external_metrics_per_sample, index=ids)

        return external_metrics_df
