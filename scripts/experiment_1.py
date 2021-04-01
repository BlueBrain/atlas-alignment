import pathlib

import numpy as np
import pandas as pd

from atlalign.base import DisplacementField
from atlalign.data import (
    annotation_volume,
    manual_registration,
    nissl_volume,
    segmentation_collapsing_labels,
)
from atlalign.metrics import evaluate_single
from atlalign.ml_utils import load_model

# Define all relevant paths
cache_dir = pathlib.Path.home() / ".atlalign"

annotation_path = cache_dir / "annotation.npy"
h5_path = cache_dir / "manual_registration.h5"
labels_path = cache_dir / "annotation_hierarchy.json"
model_local_path = cache_dir / "local.h5"
model_global_path = cache_dir / "global.h5"
nissl_path = cache_dir / "nissl.npy"


# Validation dataset definition
validation_ixs = [
    2,
    7,
    15,
    21,
    25,
    28,
    37,
    40,
    49,
    50,
    54,
    55,
    56,
    58,
    60,
    62,
    64,
    66,
    78,
    85,
    93,
    97,
    121,
    123,
    125,
    127,
    129,
    130,
    133,
    137,
    140,
    143,
    144,
    154,
    169,
    175,
    183,
    187,
    192,
    204,
    206,
    214,
    216,
    219,
    223,
    225,
    226,
    248,
    251,
    252,
    254,
    260,
    271,
    276,
]


manual_labels = manual_registration(h5_path)
nissl = nissl_volume(nissl_path)
annotation = annotation_volume(annotation_path)
labels = segmentation_collapsing_labels(labels_path)


validation_set = {}
keys = manual_labels.keys()

for val_ix in validation_ixs:
    validation_set[val_ix] = {}
    for key in keys:
        validation_set[val_ix][key] = manual_labels[key][val_ix]


model_local = load_model(model_local_path)
model_global = load_model(model_global_path)


metrics = {}
df_locals = {}

for k, data in validation_set.items():
    print(k)

    # Preparation
    img_mov = data["img"] / 255
    p = data["p"]
    section_num = p // 25
    img_ref = nissl[section_num][..., 0]

    # Global model
    inp_global = np.stack([img_ref, img_mov], axis=-1)[None, ...]
    deltas_xy_global = model_global.predict(inp_global)[1][0]
    df_global = DisplacementField(deltas_xy_global[..., 0], deltas_xy_global[..., 1])

    # Local model
    inp_local = np.stack([img_ref, df_global.warp(img_mov)], axis=-1)[None, ...]
    deltas_xy_local = model_local.predict([inp_local, np.zeros_like(inp_local)])[1][0]
    df_local = DisplacementField(deltas_xy_local[..., 0], deltas_xy_local[..., 1])
    df_locals[k] = df_local

    # Overall model
    df_pred = df_local(df_global)

    deltas_true = data["deltas_xy"]
    deltas_pred = np.stack([df_pred.delta_x, df_pred.delta_y], axis=-1)

    metrics[k], _ = evaluate_single(
        deltas_true,
        deltas_pred,
        img_mov,
        p=p,
        avol=annotation,
        collapsing_labels=labels,
        deltas_pred_inv=None,
        deltas_true_inv=data["inv_deltas_xy"],
        ds_f=8,
        depths=(0, 2, 4, 6, 8),
    )
    metrics[k]["p"] = p
    metrics[k]["image_id"] = data["image_id"]


df = pd.DataFrame(metrics).transpose()

cols = ["dice_0", "dice_2", "dice_4", "dice_6", "dice_8"]
df_overview = pd.DataFrame({"mean": df[cols].mean(), "std": df[cols].std()})

print(df_overview)


# Corrupted pixels analysis - we exclude border pixels
h, w = 320, 456
margin_ud = 1
margin_lr = 6  # the network pads the sides with 0s

corrupted_pixels = np.array(
    [
        np.sum(x.jacobian[margin_ud : h - margin_ud, margin_lr : w - margin_lr] <= 0)
        for x in df_locals.values()
    ]
)

n_pixels = (h - 2 * (margin_ud)) * (w - 2 * (margin_lr))


mean = 100 * (corrupted_pixels.mean() / n_pixels)
std = 100 * (corrupted_pixels.std() / n_pixels)

print("Corrupted_pixels")
print(
    "Up and Down cuttoff: {}\nLeft and Right cutoff: {}\n".format(margin_ud, margin_lr)
)
print("Mean:\n{}\n".format(mean))
print("STD:\n{}".format(std))
