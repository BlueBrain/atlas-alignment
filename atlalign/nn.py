"""Architecture generators."""

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

import keras
import mlflow
import numpy as np
from keras.layers import (
    Conv2D,
    Cropping2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    LeakyReLU,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
)
from keras.models import Model

from atlalign.ml_utils import (
    ALL_DVF_LOSSES,
    ALL_IMAGE_LOSSES,
    Affine2DVF,
    BilinearInterpolation,
    ExtractMoving,
    NoOp,
)


def supervised_model_factory(
    start_filters=(16,),
    downsample_filters=(16, 32, 32, 32),
    middle_filters=(32,),
    upsample_filters=(32, 32, 32, 32),
    end_filters=tuple(),
    losses=("ncc_9", "grad"),
    losses_weights=(1, 0.1),
    n_gpus=1,
    optimizer="rmsprop",
    compute_inv=False,
    mlflow_log=False,
    use_lambda=False,
):
    """Create a generic supervised registration unet.

    The mini blocks going down are: pooling - convolution - activation
    The mini blocks going up are: upsampling - convolution - activation - merge - convolution - activation
    The standard block is : convolution - activation

    Notes
    -----
    If `compute_inv=False`, then:
        * Inputs:
            * stacked reference and moving images - (None, 320, 456, 2)
        * Outputs
            * registered images - (None, 320, 456, 1)
            * predicted dvfs - (None, 320, 456, 2)

    If `compute_inv=True`, then:
        * Inputs:
            * stacked reference and moving images (None, 320, 456, 2)
            * stacked moving and reference images (None, 320, 456, 2)

        * Outputs
            * registered images - (None, 320, 456, 1)
            * predicted dvfs - (None, 320, 456, 1)
            * predicted inverse dvfs - (None, 320, 456, 1)

    Parameters
    ----------
    start_filters: tuple
        The size represents the number of starting convolutions (before downsizing) and the respective
        elements are the number of filters.

    downsample_filters : tuple
        The size represents the number of downsizing convolutions and the respective elements are the number
        of filters.

    middle_filters : tuple
        The size represents the number of standard convblocks in the middle of the net (with the most downsampled
        feature representation. The respective elements are the number of filters.

    upsample_filters : tuple
        The size represents the number of upsample convblocks of the decoder part of the net. The respective elements
         are the number of filters.

    end_filters : tuple
        The size represents the number of standard convblocks in the end of the net. The respective elements are the
        number of filters.

    losses : tuple
        If `compute_inv=False` then
            * loss to apply to registered images - needs to be a key in the ALL_IMAGE_LOSSES dictionary
            * loss to apply to predicted dvfs - needs to be a key in the ALL_DVF_LOSSES dictionary

        If `compute_inv=True` then
            * loss to apply to registered images - needs to be a key in the ALL_IMAGE_LOSSES dictionary
            * loss to apply to predicted dvfs - needs to be a key in the ALL_DVF_LOSSES dictionary
            * loss to apply to predicted inverse dvfs - needs to be a key in  the all_DVFS_LOSSES dictionary

    losses_weights : tuple
        Weights for each separate loss function (again will be 3 elements if `compute_inv=True` else 2 elements).

    n_gpus : int
        Number of gpus to use.

    optimizer : str or Keras.Optimizer
        Optimizer to be used.

    compute_inv : bool
        If True then also predicting inverse transformation. This is achieved by switching the reference and moving
        in the input and creating a new keras input out of it. Note that it effects the outputs of the model.

    mlflow_log : bool
        If True, then assumes we are inside of an MLFlow run context manager and all input parameters are
        logged.

    use_lambda : bool
        If True, then network includes Lambda layers. Otherwise, not. It is advisable not to include them because
        serialization is straightforward.

    Returns
    -------
    keras.Model
        Compiled model with the desired architecture.

    See Also
    --------
    To find a custom made generator for these networks see `atlalign.io.SupervisedGenerator`.
    """
    # Predifine layers
    if mlflow_log:
        mlflow.log_params(locals())

    # extract_reference = Lambda(lambda x: x[:, :, :, :1], name='extract_reference')
    extract_moving = (
        Lambda(lambda x: x[:, :, :, 1:], name="extract_moving")
        if use_lambda
        else ExtractMoving()
    )

    def standard_convblock(x_inp, filters, kernel_size=3, alpha_LR=0.2):
        x = Conv2D(filters, kernel_size, padding="same")(x_inp)
        x = LeakyReLU(alpha_LR)(x)

        return x

    def dowsample_convblock(x_inp, filters, kernel_size=3, alpha_LR=0.2):
        x = MaxPooling2D(pool_size=(2, 2))(x_inp)
        x = standard_convblock(
            x, filters=filters, kernel_size=kernel_size, alpha_LR=alpha_LR
        )

        return x

    def upsample_convblock(x_inp, x_to_be_merged, filters, kernel_size=3, alpha_LR=0.2):
        x = UpSampling2D(size=(2, 2))(x_inp)
        x = standard_convblock(
            x, filters=filters, kernel_size=kernel_size, alpha_LR=alpha_LR
        )
        x = concatenate([x, x_to_be_merged], axis=3)
        x = standard_convblock(
            x, filters=filters, kernel_size=kernel_size, alpha_LR=alpha_LR
        )

        return x

    # CHECKS
    height, width, n_channels = (320, 448, 2)  # aftercrop shape

    if len(downsample_filters) != len(upsample_filters):
        raise ValueError(
            "The number of downsample and upsample filters needs to be the same."
        )

    n_downsamples = len(downsample_filters)

    if (height % (2 ** n_downsamples)) != 0 or (width % (2 ** n_downsamples) != 0):
        raise ValueError("Requested downsampling not possible.")

    inputs_rm = Input(shape=(320, 456, 2), name="reg_mov")
    if compute_inv:
        inputs_mr = Input(shape=(320, 456, 2), name="mov_reg")

    def fpass(inps):
        """Forward pass."""
        x = Cropping2D(cropping=(0, 4))(inps)

        for f in start_filters:
            x = standard_convblock(x, f)

        to_be_merged_list = [x]

        for i, f in enumerate(downsample_filters):
            x = dowsample_convblock(x, f)

            if i != len(downsample_filters) - 1:
                to_be_merged_list.append(x)

        for f in middle_filters:
            x = standard_convblock(x, f)

        for f, x_to_be_merged in zip(upsample_filters, reversed(to_be_merged_list)):
            x = upsample_convblock(x, x_to_be_merged, f)

        for f in end_filters:
            x = standard_convblock(x, f)

        x = Conv2D(2, 2, padding="same")(x)

        x = ZeroPadding2D((0, 4), name="dvf")(x)

        return x

    model_copy = keras.Model(inputs=inputs_rm, outputs=fpass(inputs_rm))

    dvfs = model_copy.output

    if compute_inv:
        copy_layer = (
            Lambda(lambda x: x, name="inv_dvf") if use_lambda else NoOp(name="inv_dvf")
        )
        inv_dvfs = copy_layer(model_copy(inputs_mr))

    imgs_reg = BilinearInterpolation(name="img_registered")(
        [extract_moving(inputs_rm), dvfs]
    )

    model = Model(
        inputs=[inputs_rm, inputs_mr] if compute_inv else inputs_rm,
        outputs=[imgs_reg, dvfs, inv_dvfs] if compute_inv else [imgs_reg, dvfs],
    )

    if n_gpus > 1:
        model = keras.utils.multi_gpu_model(model, n_gpus)

    if compute_inv:
        losses = [
            ALL_IMAGE_LOSSES[losses[0]],
            ALL_DVF_LOSSES[losses[1]],
            ALL_DVF_LOSSES[losses[2]],
        ]
        metrics = {
            "dvf": ALL_DVF_LOSSES["vector_distance"],
            "inv_dvf": ALL_DVF_LOSSES["vector_distance"],
        }
    else:
        losses = [ALL_IMAGE_LOSSES[losses[0]], ALL_DVF_LOSSES[losses[1]]]
        metrics = {"dvf": ALL_DVF_LOSSES["vector_distance"]}

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=list(losses_weights),
        metrics=metrics,
    )

    model.summary()

    return model


def supervised_global_model_factory(
    filters=(16, 32, 64),
    dense_layers=(10,),
    losses=("ncc_9", "grad"),
    losses_weights=(1, 0.1),
    n_gpus=1,
    optimizer="rmsprop",
    mlflow_log=False,
    use_lambda=False,
):
    """Generate a global alignment network.

    Parameters
    ----------
    filters : tuple
        Tuple of filter sizes.

    dense_layers : None or tuple
        If None then global average pooling applied (the last conv layer needs to have 6 channels). If tuple then
        represents number of nodes in each respective dense layer. Note that in the background a final layer of
        6 nodes is created.

    losses : tuple
        * loss to apply to registered images - needs to be a key in the ALL_IMAGE_LOSSES dictionary
        * loss to apply to predicted dvfs - needs to be a key in the ALL_DVF_LOSSES dictionary

    losses_weights : tuple
        Two element tuple representing the weights for each separate loss function.

    n_gpus : int
        Number of gpus to use.

    optimizer : str or Keras.Optimizer
        Optimizer to be used.

    use_lambda : bool
        If True, then network includes Lambda layers. Otherwise, not. It is advisable not to include them because
        serialization is straightforward.

    """
    if mlflow_log:
        lcls = locals()
        lcls["model_type"] = "global_only"
        mlflow.log_params(lcls)

    def standard_convblock(x_inp, filters, kernel_size=3, relu=True):
        """Create a standard convblock."""
        x = Conv2D(filters, kernel_size, padding="same", activation="relu")(x_inp)
        x = Conv2D(
            filters, kernel_size, padding="same", activation="relu" if relu else None
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        return x

    def get_initial_weights(previous_layer_size):
        """Initialize weights such that identity transformation is produced."""
        b = np.zeros((2, 3), dtype="float32")
        b[0, 0] = 1
        b[1, 1] = 1

        W = np.zeros((previous_layer_size, 6), dtype="float32")
        weights = [W, b.flatten()]

        return weights

    extract_moving = (
        Lambda(lambda x: x[:, :, :, 1:], name="extract_moving")
        if use_lambda
        else ExtractMoving()
    )

    inputs = Input((320, 456, 2))

    x = inputs

    for f in filters:
        x = standard_convblock(x, f, relu=True)

    x = Flatten()(x)

    for d in dense_layers if dense_layers is not None else []:
        x = Dense(units=d)(x)

    x = Dense(6, weights=get_initial_weights(dense_layers[-1]))(x)
    x = Reshape((2, 3))(x)

    dvfs = Affine2DVF(shape=(320, 456))(x)

    imgs_reg = BilinearInterpolation(name="img_registered")(
        [extract_moving(inputs), dvfs]
    )

    model = Model(inputs=inputs, outputs=[imgs_reg, dvfs])

    if n_gpus > 1:
        model = keras.utils.multi_gpu_model(model, n_gpus)

    losses = [ALL_IMAGE_LOSSES[losses[0]], ALL_DVF_LOSSES[losses[1]]]
    metrics = {"dvf": ALL_DVF_LOSSES["vector_distance"]}

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=list(losses_weights),
        metrics=metrics,
    )

    model.summary()

    return model
