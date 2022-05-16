import json
import pathlib
import warnings
from argparse import ArgumentParser
from os import makedirs
from os.path import join

import nrrd
import numpy as np
import scipy
from atldld.sync import DatasetDownloader
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize
from tqdm import tqdm

from atlalign.base import DisplacementField
from atlalign.non_ml import antspy_registration
from atlalign.volume import CoronalInterpolator, GappedVolume

warnings.filterwarnings("ignore")


class SagittalInterpolator:
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
        grid = np.array(range(456))
        final_volume = np.empty((*gv.shape, len(grid)))

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
                    final_volume[r, c, :] = f(grid)
                except Exception as e:
                    print(e)

        return final_volume


def download_and_align_marker(
        dataset_id, nvol, model_gl, header,
        all_sn=None, output_filename=None,
        include_expr=True,
        is_sagittal=False,
        resolution=25.0
):
    """
    Download and align coronal images of mouse brain expressing a genetic marker
    according to a provided nissl volume.
    The experiment images will be downloaded from the Allen Institute website
    according to the provided dataset id.

    Parameters:
        dataset_id: Id of the Allen experiment
        nvol: 3D numpy ndarray Nissl volume
        model_gl: Results of the global warping machine learning
        header: header for the nrrd file
        all_sn: Results of the local warping machine learning
        output_filename: Name of the file where the dataset will be stored.
        resolution: Voxel size for the nissl volume in um
    """
    is_sagitall = False  # TODO

    slice_shape = nvol.shape[1:]
    downloader = DatasetDownloader(dataset_id, include_expression=include_expr, downsample_img=2)
    downloader.fetch_metadata()
    allen_gen = downloader.run()
    all_registered = []
    all_downsampled = []
    all_expressions = []

    for (image_id, p, img, img_exp, df) in tqdm(allen_gen):
        img_preprocessed = rgb2gray(255 - img)
        if include_expr:
            expr_preprocessed = rgb2gray(img_exp)
            img_binary = (expr_preprocessed > threshold_otsu(expr_preprocessed)) * 1
            expr_preprocessed = img_binary.astype("uint8")
            all_expressions.append(df.warp(expr_preprocessed))
        all_registered.append(df.warp(img_preprocessed))
        all_downsampled.append(resize(img_preprocessed, slice_shape))
        if not use_manual:
            all_sn.append(int(p // resolution))
    if is_sagittal:
        x_shape = nvol.shape[2]
    else:
        x_shape = nvol.shape[0]
    for i, sn in enumerate(all_sn):
        if sn >= x_shape:
            all_sn[i] = x_shape - len(all_sn) + i
    if not is_sagittal:
        # Prepare input
        inputs = np.concatenate(
            [nvol[all_sn][..., np.newaxis], np.array(all_registered)[..., np.newaxis]],
            axis=-1,
        )

        # Forward pass
        _, deltas_xy = model_gl.predict(inputs)
        # Warp the moving images
        all_dl = [
            DisplacementField(deltas_xy[i, ..., 0], deltas_xy[i, ..., 1]).warp(img_mov)
            for i, img_mov in enumerate(all_registered)
        ]
    else:
        all_dl = np.copy(all_registered)
        tot_sn = np.copy(all_sn).tolist()
        for sn, img_mov in zip(tot_sn, all_registered):
            if sn < nvol.shape[2] // 2:
                if (sn + nvol.shape[2] // 2) not in all_sn:
                    all_sn.append(sn + nvol.shape[2] // 2)
                    all_dl = np.vstack((all_dl, np.copy(img_mov)[None, :, :]))
            else:
                if (nvol.shape[2] - sn) not in all_sn:
                    all_sn.append(nvol.shape[2] - sn)
                    all_dl = np.vstack((all_dl, np.copy(img_mov)[None, :, :]))

    all_ib = []
    for i, (img_mov, sn) in tqdm(enumerate(zip(all_dl, all_sn))):
        if is_sagittal:
            df, _ = antspy_registration(nvol[:, :, sn], img_mov)
        else:
            df, _ = antspy_registration(nvol[sn], img_mov)
        if include_expr:
            all_ib.append(df.warp(all_expressions[i]))
        else:
            all_ib.append(df.warp(img_mov))

    gv = GappedVolume(all_sn, all_ib)

    if is_sagittal:
        ci = SagittalInterpolator(kind="linear")
    else:
        ci = CoronalInterpolator(kind="linear")
    final_volume = ci.interpolate(gv)

    return final_volume


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "nissl_path",
        type=pathlib.Path,
        help="Path to the Nissl volume."
    )
    parser.add_argument(
        "local_model_path",
        type=pathlib.Path,
        help="Path to the local deep learning model."
    )
    parser.add_argument(
        "global_model_path",
        type=pathlib.Path,
        help="Path to the global deep learning model."
    )
    parser.add_argument(
        "genes",
        type=str,
        help="Comma separated list of gene ids to download and align."
    )
    parser.add_argument(
        "output_path",
        type=pathlib.Path,
        help="Path to the folder where the results will be stored."
    )
    parser.add_argument(
        "-e",
        "--include-expression",
        action="store_true",
        help="If True, we also download and align expression images."
    )
    args = parser.parse_args()

    # imports
    from unittest.mock import Mock

    from atlalign.ml_utils import load_model, merge_global_local

    print("Aligning markers images to the Nissl volume.")

    nvol, header = nrrd.read(args.nissl_path)
    nvol = nvol / nvol.max()

    genelist = args.genes.split(",")

    local_model = load_model(args.local_model_path)
    global_model = load_model(args.global_model_path)
    model_gl = merge_global_local(global_model, local_model)

    args.output_path.mkdir(exist_ok=True, parents=True)

    for dataset_id in genelist:
        print(f"Downloading and aligning {dataset_id=}")
        # temp
        download_and_align_marker = Mock(return_value=np.zeros((528, 320, 456)))
        volume = download_and_align_marker(
            dataset_id,
            nvol,
            model_gl,
            include_expr=args.include_expression,
        )

        gene_folder = args.output_path / dataset_id
        gene_folder.mkdir(exist_ok=True, parents=True)

        volume_path = gene_folder / "volume.nrrd"

        nrrd.write(str(volume_path), volume, header=header)
