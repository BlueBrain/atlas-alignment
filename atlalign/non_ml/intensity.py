"""Collection of intensity based registration methods."""

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
import warnings

warnings.simplefilter(action="ignore")  # noqa

import ants  # noqa
import nibabel as nib  # noqa

from atlalign.base import GLOBAL_CACHE_FOLDER, DisplacementField  # noqa


def antspy_registration(
    fixed_img,
    moving_img,
    registration_type="SyN",
    reg_iterations=(40, 20, 0),
    aff_metric="mattes",
    syn_metric="mattes",
    verbose=False,
    initial_transform=None,
    path=GLOBAL_CACHE_FOLDER,
):
    """Register images using ANTsPY.

    Parameters
    ----------
    fixed_img: np.ndarray
        Fixed image.

    moving_img: np.ndarray
        Moving image to register.

    registration_type: {'Translation', 'Rigid', 'Similarity', 'QuickRigid', 'DenseRigid', 'BOLDRigid', 'Affine',
                        'AffineFast', 'BOLDAffine', 'TRSAA', 'ElasticSyN', 'SyN', 'SyNRA', 'SyNOnly', 'SyNCC', 'SyNabp',
                        'SyNBold', 'SyNBoldAff', 'SyNAggro', 'TVMSQ', 'TVMSQC'}, default 'SyN'

        Optimization algorithm to use to register (more info: https://antspy.readthedocs.io/en/latest/registration.
        html?highlight=registration#ants.registration)

    reg_iterations: tuple, default (40, 20, 0)
        Vector of iterations for SyN.

    aff_metric: {'GC', 'mattes', 'meansquares'}, default 'mattes'
        The metric for the affine part.

    syn_metric: {'CC', 'mattes', 'meansquares', 'demons'}, default 'mattes'
        The metric for the SyN part.

    verbose : bool, default False
        If True, then the inner solver prints convergence related information in standard output.

    path : str
        Path to a folder to where to save the `.nii.gz` file representing the composite transform.

    initial_transform : list or None
        Transforms to prepend the before the registration.

    Returns
    -------
    df: DisplacementField
        Displacement field between the moving and the fixed image

    meta : dict
        Contains relevant images and paths.

    """
    path = str(path)
    path += "" if path[-1] == "/" else "/"

    fixed_ants_image = ants.image_clone(ants.from_numpy(fixed_img), pixeltype="float")
    moving_ants_image = ants.image_clone(ants.from_numpy(moving_img), pixeltype="float")
    meta = ants.registration(
        fixed_ants_image,
        moving_ants_image,
        registration_type,
        reg_iterations=reg_iterations,
        aff_metric=aff_metric,
        syn_metric=syn_metric,
        verbose=verbose,
        initial_transform=initial_transform,
        syn_sampling=32,
        aff_sampling=32,
    )

    filename = ants.apply_transforms(
        fixed_ants_image,
        moving_ants_image,
        meta["fwdtransforms"],
        compose=path + "final_transform",
    )

    df = nib.load(filename)
    data = df.get_fdata()
    data = data.squeeze()
    dx = data[:, :, 1]
    dy = data[:, :, 0]
    df_final = DisplacementField(dx, dy)

    return df_final, meta
