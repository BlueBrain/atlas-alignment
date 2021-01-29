"""A collection of utils for all visualization scripts."""

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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation

from atlalign.base import DisplacementField
from atlalign.metrics import angular_error_of


def create_animation(
    df,
    img,
    frames_per_second=30,
    n_seconds=3,
    repeat=False,
    blit=False,
    cmap="gray",
    img_ref=None,
    n_ref=3,
    duration_ref=1,
):
    """Create a slow motion animation of a warping.

    Parameters
    ----------
    df : DisplacementField or list
        If an instance of the DisplacementField class representing then representing a single transformation. If a
        list of DisplacementField instances then represents a pipeline of different transformations to be applied
        in the respective order.

    img : np.ndarray
        Image to be warped. Needs to have the same shape as teh `df` and dtype either
        uint8 or float32.

    frames_per_second : int, default 30
        Number of frames per second.

    n_seconds : int, default 3
        Number of seconds one df will last. Total number of seconds is `len(df) * n_seconds`.

    repeat : bool
        If True, animation is automatically restarted.

    blit : bool
        Controls whether blitting is used to optimize drawing.

    cmap : str, default 'gray'
        Only applicable if image grayscale.

    img_ref : np.array or None
        If supplied then at the end of the animation switch between moving and registered `n_ref` of times
        where each blit lasts `duration_ref` seconds.

    n_ref : int
        Number of times to switch between `img_ref` and registered image. Only active when `img_reg` is not
        None.

    duration_ref : int
        Number of seconds `img_ref` is visible per blit.

    Returns
    -------
    ani : matplotlib.animation.ArtistAnimation
        Animation object that can be viewed in a jupter notebook for example.


    Notes
    -----
    To make it viewable in a jupyter notebook one needs to do the following

    >>> from matplotlib import  rc
    >>> rc('animation', html='jshtml')

    If you get errors using these settings consider replacing `html='jshtml'`
    by `html='html5'` above.

    Additionally, it is necessary to install ffpmeg package. On Ubuntu this can be done:

    ```bash
    sudo apt install ffmpeg
    ```

    """
    # Prepare variables
    total_frames = n_seconds * frames_per_second  # total frames per one df
    interval = int(1000 / frames_per_second)
    all_frames = []

    df_list = df if isinstance(df, list) else [df]

    # Prepare plot
    fig = plt.figure()
    plt.axis("off")

    # Collect frames
    for df_ in df_list:
        for i in range(total_frames + 1):
            df_temp = df_ * (i / total_frames)
            warped_img_ = df_temp.warp(img)
            warped_img = plt.imshow(warped_img_, cmap=cmap)
            all_frames.append([warped_img])

        # Update starting image with the last warped image
        img = warped_img_

    if img_ref is not None:
        img_mov_axes = all_frames[-1][0]
        img_ref_axes = plt.imshow(img_ref, cmap=cmap)
        for i in range(n_ref):
            # reference
            all_frames.extend(int(frames_per_second * duration_ref) * [[img_ref_axes]])
            # moving
            all_frames.extend(int(frames_per_second * duration_ref) * [[img_mov_axes]])

    ani = animation.ArtistAnimation(
        fig, all_frames, interval=interval, blit=blit, repeat=repeat, repeat_delay=None
    )

    return ani


def create_grid(shape, grid_spacing=20, grid_thickness=3):
    """Create a grid to see warpings clearly.

    Parameters
    ----------
    shape : tuple
        Tuple of (height, width) which represent the shape of the output image.

    grid_spacing : int
        Both horizontal and vertical spacing of consecutive lines.

    grid_thickness : int
        Thickness of all lines.

    Returns
    -------
    img_grid : np.ndarray
        An image of the grid.

    """
    grid_shape = (grid_h, grid_w) = shape[:2]
    grid = np.zeros(grid_shape)

    # Populate horizontal
    for c in range(0, grid_w, grid_spacing):
        grid[:, c : c + grid_thickness] = 255

    # Populate vertical
    for r in range(0, grid_h, grid_spacing):
        grid[r : r + grid_thickness, :] = 255

    grid = 255 - grid

    return grid


def create_segmentation_image(segmentation_array, colors_dict=None):
    """Turn segmentation array into a colorful image.

    Parameters
    ----------
    segmentation_array : np.array
        An array of shape (h, w) and dtype ``int`` where each number represents a unique class.

    colors_dict : None or dict
        If None, then all classes are assigned a random color (except for 0 which by default gets a black color).
        If dict, keys are integers representing classes and values are tuples of size 3 representing (R, G, B). If
        a class is not contained in the dict then color randomly generated.


    Returns
    -------
    segmentation_img : np.array
        An image of shape (h, w) and dtype `uint8`` and 3 channels (RGB).

    colors_dict : dict
        Color (values) per class (keys) dictionary. If no `colors_dict` passed then a new instance. If passed,
        then it is an updated version.

    """
    if not np.issubdtype(segmentation_array.dtype, np.integer):
        raise TypeError("Only integer valued classes are allowed.")

    if colors_dict is None:
        colors_dict = {0: np.array([0, 0, 0])}  # background

    all_labels = np.unique(segmentation_array)

    segmentation_img = np.zeros((*segmentation_array.shape, 3), dtype=np.uint8)

    for lb in all_labels:

        if lb not in colors_dict:
            color = np.random.randint(255, size=3)
            colors_dict[lb] = color
        else:
            color = colors_dict[lb]

        segmentation_img[segmentation_array == lb] = color

    return segmentation_img, colors_dict


def generate_df_plots(df_true, df_pred, filepath=None, figsize=(15, 15)):
    """Generate displacement vector plots.

    df_true : DisplacementField
        Truth. Assumes that shape is (320, 456).

    df_pred : DisplacementField
        Prediction. Assumes that shape is (320, 456)

    filepath : None or pathlib.Path
        If specified, then the path to where the figure saved as a PNG image.
        If not specified, then shown.
    """
    plt.ioff()
    fig, (
        (ax_norm, ax_norm_p),
        (ax_angle, ax_angle_p),
        (ax_jacob, ax_jacob_p),
        (ax_grid, ax_grid_p),
    ) = plt.subplots(4, 2, figsize=figsize)

    df_base = DisplacementField.generate(
        (320, 456), approach="affine_simple", translation_x=1
    )  # make the angle work

    bar_norm = fig.add_axes([0.95, 0.772, 0.03, 0.2])
    bar_angle = fig.add_axes([0.95, 0.525, 0.03, 0.2])
    bar_jacob = fig.add_axes([0.95, 0.275, 0.03, 0.2])

    # Jacobian
    jacob_true = df_true.jacobian
    jacob_pred = df_pred.jacobian
    jacob_vmin, jacob_vmax = min(jacob_true.min(), jacob_pred.min()), max(
        jacob_true.max(), jacob_pred.max()
    )
    ax_jacob.set_axis_off()
    sns.heatmap(
        jacob_true,
        ax=ax_jacob,
        cbar_ax=bar_jacob,
        center=0,
        cmap="seismic_r",
        vmin=jacob_vmin,
        vmax=jacob_vmax,
    )
    ax_jacob.set_title("Jacobian - Ground Truth")

    ax_jacob_p.set_axis_off()
    sns.heatmap(
        jacob_pred,
        ax=ax_jacob_p,
        cbar_ax=bar_jacob,
        center=0,
        cmap="seismic_r",
        vmin=jacob_vmin,
        vmax=jacob_vmax,
    )
    ax_jacob_p.set_title("Jacobian - Predicted")

    # GRID
    img_grid = create_grid((320, 456))
    img_grid_warped = df_true.warp(img_grid)
    ax_grid.set_axis_off()
    ax_grid.imshow(img_grid_warped, cmap="gray")
    ax_grid.set_title("Warped Grid - Ground Truth")

    img_grid_warped_p = df_pred.warp(img_grid)
    ax_grid_p.set_axis_off()
    ax_grid_p.imshow(img_grid_warped_p, cmap="gray")
    ax_grid_p.set_title("Warped Grid - Predicted")

    # NORM
    norm_vmin, norm_vmax = 0, max(df_true.norm.max(), df_pred.norm.max())
    ax_norm.set_axis_off()
    sns.heatmap(
        df_true.norm, ax=ax_norm, cbar_ax=bar_norm, vmin=norm_vmin, vmax=norm_vmax
    )
    ax_norm.set_title("Norm - Ground Truth")

    ax_norm_p.set_axis_off()
    sns.heatmap(
        df_pred.norm, ax=ax_norm_p, cbar_ax=bar_norm, vmin=norm_vmin, vmax=norm_vmax
    )
    ax_norm_p.set_title("Norm - Predicted")

    # Angle
    angle_vmin, angle_vmax = 0, 180
    ax_angle.set_title("Angle wrt positive x-axis - Ground Truth")
    ax_angle.set_axis_off()
    angles = angular_error_of([df_true], [df_base])[1]
    sns.heatmap(
        angles,
        ax=ax_angle,
        cbar_ax=bar_angle,
        mask=~np.isfinite(angles),
        cmap="hot_r",
        vmin=angle_vmin,
        vmax=angle_vmax,
    )

    ax_angle_p.set_title("Angle wrt positive x-axis - Predicted")
    ax_angle_p.set_axis_off()
    angles = angular_error_of([df_pred], [df_base])[1]
    sns.heatmap(
        angles,
        ax=ax_angle_p,
        cbar_ax=bar_angle,
        mask=~np.isfinite(angles),
        cmap="hot_r",
        vmin=angle_vmin,
        vmax=angle_vmax,
    )
    # fig.tight_layout(rect=[0, 0, .95, 1])

    if filepath is not None:
        fig.savefig(str(filepath))
    else:
        plt.show()


def chain_predict(model, inp, n_iterations=1):
    """Run alignment recursively.

    Parameters
    ----------
    model : keras.models.Model
        A trained model that whose inputs have shape (batch_size, h, w, 2) - last dimension represents
        stacking of atlas and input image. The outputs are of the same shape where the last dimension represents
        stacking of delta_x and delta_y of the displacement field.

    inp : np.ndarray
        An array of shape (h, w, 2) or (1, h, w, 2) representing the atlas and input image.

    Returns
    -------
    unwarped_img_list : list
        List of np.ndarrays of shape (h, w) representign the unwarped image at each iteration.

    """
    # Checks
    if inp.ndim == 3:
        inp_ = np.array([inp])

    elif inp.ndim == 4 and inp.shape[0] == 1:
        inp_ = inp

    else:
        raise ValueError("Input has incorrect shape of {}".format(inp.shape))

    shape = inp.shape[1:3]

    df = DisplacementField.generate(shape, approach="identity")

    img_atlas = inp_[0, :, :, 0]
    img_warped = inp_[0, :, :, 1]

    unwarped_img_list = [img_warped]

    for i in range(n_iterations):
        new_inputs = np.concatenate(
            (
                img_atlas[np.newaxis, :, :, np.newaxis],
                unwarped_img_list[-1][np.newaxis, :, :, np.newaxis],
            ),
            axis=3,
        )

        pred = model.predict(new_inputs)

        delta_x_pred = pred[0, ..., 0]
        delta_y_pred = pred[0, ..., 1]

        df_pred = DisplacementField(delta_x_pred, delta_y_pred)

        df_pred_inv = df_pred.pseudo_inverse(ds_f=8)

        df = df_pred_inv(df).adjust()
        img_unwarped_pred = df.warp(img_warped)

        unwarped_img_list.append(img_unwarped_pred)

    return unwarped_img_list
