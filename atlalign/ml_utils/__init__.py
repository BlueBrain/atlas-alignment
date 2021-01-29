"""Prevent 3 level imports for important objects."""

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

from atlalign.ml_utils.augmentation import augmenter_1  # noqa
from atlalign.ml_utils.callbacks import MLFlowCallback, get_mlflow_artifact_path  # noqa
from atlalign.ml_utils.io import SupervisedGenerator  # noqa
from atlalign.ml_utils.layers import (  # noqa
    Affine2DVF,
    BilinearInterpolation,
    DVFComposition,
    ExtractMoving,
    NoOp,
    block_stn,
)
from atlalign.ml_utils.losses import (  # noqa
    DVF2IMG,
    NCC,
    Grad,
    Mixer,
    PerceptualLoss,
    VDClipper,
    cross_correlation,
    jacobian,
    jacobian_distance,
    mse_po,
    psnr,
    ssim,
    vector_distance,
)
from atlalign.ml_utils.models import (  # noqa
    load_model,
    merge_global_local,
    replace_lambda_in_config,
    save_model,
)

# Create utility dictionary
ALL_IMAGE_LOSSES = {
    "mae": "mae",
    "mse": "mse",
    "ncc_5": NCC(win=5).loss,
    "ncc_9": NCC(win=9).loss,
    "ncc_12": NCC(win=12).loss,
    "ncc_20": NCC(win=20).loss,
    "pearson": cross_correlation,
    "perceptual_loss_net-lin_alex": PerceptualLoss(model="net-lin", net="alex").loss,
    "perceptual_loss_net-lin_vgg": PerceptualLoss(model="net-lin", net="vgg").loss,
    "perceptual_loss_net_alex": PerceptualLoss(model="net", net="alex").loss,
    "perceptual_loss_net_vgg": PerceptualLoss(model="net", net="vgg").loss,
    "psnr": psnr,
    "ssim": ssim,
}
ALL_DVF_LOSSES = {
    "grad": Grad().loss,
    "jacobian": jacobian,
    "jacobian_distance": jacobian_distance,
    "mae": "mae",
    "mse": "mse",
    "mse_po": mse_po,
    "vector_distance": vector_distance,
    "vdclip2": VDClipper(20, power=2).loss,
    "vdclip3": VDClipper(20, power=3).loss,
    "vector_jacobian_distance": Mixer(vector_distance, jacobian_distance).loss,
    "vector_jacobian_distance_02": Mixer(
        vector_distance, jacobian_distance, weights=[0.2, 0.8]
    ).loss,
}

ALL_DVF_LOSSES = {
    **ALL_DVF_LOSSES,
    **{k: DVF2IMG(v).loss for k, v in ALL_IMAGE_LOSSES.items() if callable(v)},
}

all_dvf_losses_items = list(ALL_DVF_LOSSES.items())
all_dvf_losses_items.sort(key=lambda x: x[0])

MIXED_DVF_LOSSES = {
    "{}&{}".format(k_i, k_o): Mixer(v_i, v_o).loss
    for i, (k_o, v_o) in enumerate(all_dvf_losses_items)
    for j, (k_i, v_i) in enumerate(all_dvf_losses_items)
    if j < i and callable(v_i) and callable(v_o)
}

ALL_DVF_LOSSES = {**ALL_DVF_LOSSES, **MIXED_DVF_LOSSES}
