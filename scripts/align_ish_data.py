import warnings
import nrrd
import json
import numpy as np
from os import makedirs
from os.path import join
from tqdm import tqdm
from argparse import ArgumentParser
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize
from atldld.sync import DatasetDownloader
from atlalign.base import DisplacementField
from atlalign.ml_utils import merge_global_local, load_model
from atlalign.non_ml import antspy_registration
from atlalign.volume import GappedVolume, CoronalInterpolator
import scipy

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


def align_marker(
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

    use_manual = True
    if all_sn is None:
        all_sn = []
        use_manual = False
    if output_filename is None:
        output_filename = str(dataset_id) + "_expr.nrrd"

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
    nrrd.write(output_filename, final_volume, header=header)
    return all_sn


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ccf")
    args = parser.parse_args()
    print("Aligning markers images to the Nissl volume.")

    DATA_FOLDER = "../data/"
    CCF_version = args.ccf

    dl_global_path = join(DATA_FOLDER, "atlalign/global/boring_bear/")
    dl_local_path = join(DATA_FOLDER, "atlalign/local/blue_bird/")
    nvol, h = nrrd.read(join(DATA_FOLDER, CCF_version, "ara_nissl_25.nrrd"))
    nvol = nvol / nvol.max()

    # For each dataset, every coronal section was manually realigned
    # to the corresponding anatomical section of the high-resolution Nissl-stained mouse brain.
    # reordered_slices = json.loads(
    #     open(join(DATA_FOLDER, "realigned_slices.json"), "r").read()
    # )
    reordered_slices = {}

    # List of ISH experiment ids according to the AIBS API
    # ish experiments used in Rodarie et al. 2022
    genelist = [868, 1001, 77371835, 479]
    namelist = ["pvalb", "SST", "VIP", "gad1"]
    is_sagittal = [False for _ in range(4)]
    take_expression = [True for _ in range(4)]

    genelist += [79556706]
    namelist += ["gad1_bis"]
    is_sagittal += [False]
    take_expression += [True]

    # ish experiments used in Erö et al. 2018
    genelist += [79591671, 112202838, 79591593, 1175, "75147760"]
    namelist += ["GFAP", "MBP", "S100b", "CNP", "NRN1"]
    is_sagittal += [False for _ in range(5)]
    take_expression += [False for _ in range(5)]

    genelist += [68161453, 68631984 ]
    namelist += ["TMEM119", "ALDH1L1"]
    is_sagittal += [True, True]
    take_expression += [False, False]

    model_gl = merge_global_local(load_model(dl_global_path), load_model(dl_local_path))
    makedirs(join(DATA_FOLDER, CCF_version, "marker_volumes"), exist_ok=True)
    for i_dataset, dataset_id in enumerate(genelist):
        print("Aligning dataset to nissl for " + namelist[i_dataset])
        print("Expr: " + str(take_expression[i_dataset]) + ", Sagittal: " + str(is_sagittal[i_dataset]))
        if str(dataset_id) not in reordered_slices:
            all_sn = None
        else:
            all_sn = np.array(reordered_slices[str(dataset_id)])
        reordered_slices[str(dataset_id)] = align_marker(dataset_id, nvol, model_gl,
                                                         h, all_sn,
                                                         join(DATA_FOLDER, CCF_version,
                                                              "marker_volumes", namelist[i_dataset] + ".nrrd"),
                                                         is_sagittal=is_sagittal[i_dataset],
                                                         include_expr=take_expression[i_dataset]
                                                         )
    with open(join(DATA_FOLDER, CCF_version, "marker_volumes", "realigned_slices.json"), 'w') as fp:
        json.dump(reordered_slices, fp, indent=4)
