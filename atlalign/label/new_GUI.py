"""Graphical User Interface for manual registration."""

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

import json
import os
import pickle
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider
from skimage.color import gray2rgb
from skimage.exposure import equalize_hist
from skimage.util import img_as_float32

from atlalign.base import DisplacementField
from atlalign.data import nissl_volume
from atlalign.visualization import create_grid


def run_GUI(img_ref, img_mov, mode="ref2mov", title=""):
    """Graphical user interface for manual labeling.

    Notes
    -----
    If `mode` == 'ref2mov' then one first specifies the point in the reference image (circle marker) and then the
    corresponding pixel in the moving image (star marker). Note that these pairs have the same color. To delete a
    specific pair hover above on the undesirable reference point and press space bar and this will automatically delete
    it. Note that deletion delete both the reference point and the moving point but you can only point at the
    reference one for deletions.

    Parameters
    ----------
    img_ref : np.ndaray
        Reference image. Needs to be dtype == np.uint8.

    img_mov : np.ndarray
        Input image. Needs to be dtype == np.uint8 and the same shape as `img_ref`.

    mode : str, {'ref2mov', 'mov2ref'}
        If 'ref2mov' then the first point should be in the reference image and the other point in the moving one.
        For 'mov2ref' its vice versa.

    title : str,
        Additional title of the figure.

    Returns
    -------
    df : DisplacementField
        Displacement field corresponding to the last change before closing the window of the GUI.

    keypoints : dict
        Dictionary of keypoints.

    img_reg : np.ndarray
        Registered image.

    interpolation_method : str
        Interpolation method

    kernel : str
        Kernel.

    """
    if not img_ref.shape == img_mov.shape:
        raise ValueError("The fixed and moving image need to have the same shape.")

    if not (img_ref.dtype == np.float32 and img_mov.dtype == np.float32):
        raise TypeError("Only works with float32 dtype")

    if mode not in {"ref2mov", "mov2ref"}:
        raise ValueError("The mode can only be ref2mov or mov2ref.")

    class HelperGlobal:
        """Just a way how to avoid using global variables.

        Attributes
        ----------
        img_ref_ : np.ndarray
            Copy of the reference image.

        img_mov_ : np.ndarray
            Copy of the moving image.

        img_reg : np.ndarray
            Continuously updated registered image.

        ax, ax_reg : matplotlib.Axes
            Axes objects:
                ax — overlay of the `img_ref_` and `img_mov_` together with scatter of current keypoints.
                ax_reg - overlay of the `img_ref_ and img_reg`.

        keypoints : dict
            The keys are (x_ref, y_ref) pairs in the reference image whereas the values are (x_mov, y_mov)
            tuples. Note that we make heavy use of the None sentinel whenever a new pair is being inputted.
            If `mode` = 'ref2mov' then it holds that n_ref = n_mov or n_ref = n_mov + 1. In the second case,
            the sentinel is used in dictionary values. If `mode` = 'mov2ref' then it holds that n_ref = n_mov or
            n_ref = n_mov - 1. In the second case, the sentinel is used in dictionary keys.

        all_colors : deque
            All possible colors for the scatter plot. Note that we infinitely iteratre through this
            for new input points.

        epsilon : int
            A parameter that determines the rectangle around a clicked point during deletions (spacebar)
            The higher this parameter the less precise you need to be when trying to delete a reference keypoint.

        """

        def __init__(self):

            # Save internally original images (immutable)
            self.img_ref_ = img_ref.copy()
            self.img_mov_ = img_mov.copy()

            #
            self.img_reg = img_mov.copy()  # continuously_updated
            self.df = DisplacementField.generate(
                self.img_ref_.shape, approach="identity"
            )  # latest DVF
            self.grid_ = create_grid(
                shape=self.img_ref_.shape, grid_spacing=15, grid_thickness=2
            )  # the unwarped grid

            # dummy
            self.img_dummy = np.zeros(self.img_ref_.shape)

            # Hyperparameters
            self.colormaps = ["hot", "gray", "cool", "viridis", "spring"]

            self.cmap_ref = "gray"  # needs index in the colormaps list
            self.cmap_movreg = "hot"  # needs index in the colormaps list

            self.th_ref = 10 / 255
            self.th_movreg = 100 / 255

            self.ref_first = True
            self.show_grid = False
            self.show_arrows = True

            self.alpha_ref = 0.8
            self.alpha_movreg = 0.5

            # interpolation related
            self.interpolation_methods = ["griddata", "rbf"]
            self.interpolation_method = "rbf"
            self.interpolation_method_prev = self.interpolation_method

            self.kernels = [
                "multiquadric",
                "inverse",
                "gaussian",
                "linear",
                "cubic",
                "quintic",
                "thin_plate",
            ]
            self.kernel = "thin_plate"
            self.kernel_prev = self.kernel

            # Visual
            self.marker_ref = "o"
            self.marker_mov = "*"
            self.marker_size_ref = 7 ** 2
            self.marker_size_mov = 11 ** 2

            self.all_colors = deque(
                ["red", "green", "blue", "yellow", "orange", "pink", "brown", "cyan"]
            )  # left start
            # self.all_colors = deque([cm.tab20(i / 100) for i in range(0, 100, 5)])

            # Axis
            self.fig, (self.ax, self.ax_reg) = plt.subplots(1, 2, figsize=(20, 20))
            # self.fig.canvas.set_window_title(title)
            self.fig.tight_layout()
            self.fig.suptitle(title, fontsize=17)

            self.ax.set_axis_off()
            self.ax_reg.set_axis_off()

            self._define_widgets()

            # Attributes
            self.keypoints = {}  # # (x_ref, y_ref) -> (x_inp, y_inp)
            self.keypoints_prev = {}
            self.colors = {}  # (x_ref, y_ref) -> color ()

            self.epsilon = 3

            # Initilize p
            self._update_plots()

        def _define_widgets(self):
            """Define all widgets."""
            # toggle reset
            self.reset_button = Button(
                plt.axes([0.20, 0.01, 0.1, 0.03]),
                "Reset",
                # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
                hovercolor=None,
            )

            def on_clicked(*args, **kwargs):
                self.keypoints = {}
                self._update_plots()

            self.reset_button.on_clicked(on_clicked)

            # toggle ref_first
            self.ref_first_button = Button(
                plt.axes([0.4, 0.01, 0.1, 0.03]),
                "Change order",
                # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
                hovercolor=None,
            )

            def on_clicked(*args, **kwargs):
                self.ref_first = not self.ref_first
                self._update_plots()

            self.ref_first_button.on_clicked(on_clicked)

            # toggle show_arrows
            self.show_arrows_button = Button(
                plt.axes([0.6, 0.01, 0.1, 0.03]),
                "Show arrows",
                # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
                hovercolor=None,
            )

            def on_clicked(*args, **kwargs):
                self.show_arrows = not self.show_arrows
                self._update_plots()

            self.show_arrows_button.on_clicked(on_clicked)

            # toggle show_grid
            self.show_grid_button = Button(
                plt.axes([0.8, 0.01, 0.1, 0.03]),
                "Show grid",
                # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
                hovercolor=None,
            )

            def on_clicked(*args, **kwargs):
                self.show_grid = not self.show_grid
                self._update_plots()

            self.show_grid_button.on_clicked(on_clicked)

            # threshold ref
            axcolor = "lightgoldenrodyellow"
            self.th_ref_slider = Slider(
                plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor),
                "Reference Threshold",
                0.0,
                1,
                valinit=self.th_ref,
                color=[0.0, 1.0, 0.0],
            )

            def on_changed(val):
                self.th_ref = val
                self._update_plots()

            self.th_ref_slider.on_changed(on_changed)

            # threshold movreg
            axcolor = "lightgoldenrodyellow"
            self.th_movreg_slider = Slider(
                plt.axes([0.25, 0.075, 0.65, 0.03], facecolor=axcolor),
                "Moving/Registered Threshold",
                0.0,
                1,
                valinit=self.th_movreg,
                color=[0.0, 1.0, 0.0],
            )

            def on_changed(val):
                self.th_movreg = val
                self._update_plots()

            self.th_movreg_slider.on_changed(on_changed)

            # alpha ref
            axcolor = "lightgoldenrodyellow"
            self.alpha_ref_slider = Slider(
                plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor),
                "Alpha Ref",
                0.0,
                1,
                valinit=self.alpha_ref,
                color=[0.0, 0.0, 1.0],
            )

            def on_changed(val):
                self.alpha_ref = val
                self._update_plots()

            self.alpha_ref_slider.on_changed(on_changed)

            # alpha movreg
            axcolor = "lightgoldenrodyellow"
            self.alpha_movreg_slider = Slider(
                plt.axes([0.25, 0.125, 0.65, 0.03], facecolor=axcolor),
                "Alpha Movinig/Registered",
                0.0,
                1,
                valinit=self.alpha_movreg,
                color=[0.0, 0.0, 1.0],
            )

            def on_changed(val):
                self.alpha_movreg = val
                self._update_plots()

            self.alpha_movreg_slider.on_changed(on_changed)

            # cmap ref

            axcolor = "lightgoldenrodyellow"
            rax = plt.axes([0.1, 0.85, 0.05, 0.10], facecolor=axcolor)
            rax.set_title("Ref")

            self.cmap_ref_radio = RadioButtons(
                rax, labels=self.colormaps, active=self.colormaps.index(self.cmap_ref)
            )

            def on_clicked(*args, **kwargs):
                self.cmap_ref = self.cmap_ref_radio.value_selected
                self._update_plots()

            self.cmap_ref_radio.on_clicked(on_clicked)

            # cmap_movreg
            axcolor = "lightgoldenrodyellow"
            rax = plt.axes([0.05, 0.85, 0.05, 0.10], facecolor=axcolor)
            rax.set_title("Mov/Reg")
            self.cmap_movreg_radio = RadioButtons(
                rax,
                labels=self.colormaps,
                active=self.colormaps.index(self.cmap_movreg),
            )

            def on_clicked(*args, **kwargs):
                self.cmap_movreg = self.cmap_movreg_radio.value_selected
                self._update_plots()

            self.cmap_movreg_radio.on_clicked(on_clicked)

            # interpolation method
            axcolor = "lightgoldenrodyellow"
            rax = plt.axes([0.85, 0.85, 0.05, 0.10], facecolor=axcolor)
            rax.set_title("Interpolator")
            self.interpolation_method_radio = RadioButtons(
                rax,
                labels=self.interpolation_methods,
                active=self.interpolation_methods.index(self.interpolation_method),
            )

            def on_clicked(*args, **kwargs):
                self.interpolation_method = (
                    self.interpolation_method_radio.value_selected
                )
                self._update_plots()

            self.interpolation_method_radio.on_clicked(on_clicked)

            # kernel
            axcolor = "lightgoldenrodyellow"
            rax = plt.axes([0.9, 0.85, 0.09, 0.10], facecolor=axcolor)
            rax.set_title("Kernel")
            self.kernel_radio = RadioButtons(
                rax, labels=self.kernels, active=self.kernels.index(self.kernel)
            )

            def on_clicked(*args, **kwargs):
                self.kernel = self.kernel_radio.value_selected
                self._update_plots()

            self.kernel_radio.on_clicked(on_clicked)

            # Transform quality
            axcolor = "lightgoldenrodyellow"
            self.rax_quality = plt.axes([0.45, 0.8, 0.10, 0.1], facecolor=axcolor)
            self.rax_quality.set_axis_off()

        def _update_plots(self):
            """Render the entire figure with the most recent parameters."""
            new_kps = self.keypoints != self.keypoints_prev
            new_ip = (
                self.interpolation_method != self.interpolation_method_prev
                or self.kernel != self.kernel_prev
            )
            is_complete = not np.any(
                [k is None or v is None for k, v in self.keypoints.items()]
            )

            if (new_kps and is_complete) or new_ip:
                self.keypoints_prev = deepcopy(self.keypoints)

                self.interpolation_method_prev = self.interpolation_method
                self.kernel_prev = self.kernel

                # Check if any keypoints
                if not self.keypoints:
                    self.df = DisplacementField.generate(
                        self.img_ref_.shape, approach="identity"
                    )
                    self.img_reg = self.img_mov_.copy()

                else:

                    # Interpolation preparation
                    all_kps = [
                        (k, v)
                        for k, v in self.keypoints.items()
                        if k is not None and v is not None
                    ]

                    coords_ref = [x[0] for x in all_kps]
                    coords_inp = [x[1] for x in all_kps]

                    points = np.flip(
                        np.array(coords_ref), axis=1
                    )  # control_points uses (row, col) = ( y, x)
                    values_delta_y = np.array(
                        [
                            xy_inp[1] - xy_ref[1]
                            for xy_ref, xy_inp in zip(coords_ref, coords_inp)
                        ]
                    )
                    values_delta_x = np.array(
                        [
                            xy_inp[0] - xy_ref[0]
                            for xy_ref, xy_inp in zip(coords_ref, coords_inp)
                        ]
                    )

                    interpolator_kwargs = (
                        {}
                        if self.interpolation_method == "griddata"
                        else {"function": "{}".format(self.kernel)}
                    )

                    # Actual interpolation
                    self.df = DisplacementField.generate(
                        self.img_ref_.shape,
                        approach="control_points",
                        points=points,
                        values_delta_x=values_delta_x,
                        values_delta_y=values_delta_y,
                        anchor_corners=True,
                        interpolation_method=self.interpolation_method,
                        interpolator_kwargs=interpolator_kwargs,
                    )

                    # Plot
                    # self.df.plot_dvf(ax=self.ax_df)

                    # Update warped image
                    self.img_reg = self.df.warp(self.img_mov_.copy())

            self.ax.cla()
            self.ax_reg.cla()

            n_ref = len([x for x in self.keypoints.keys() if x is not None])
            n_mov = len([x for x in self.keypoints.values() if x is not None])

            perc_good = np.sum(self.df.jacobian > 0) / np.prod(self.df.shape)
            average_disp = self.df.average_displacement

            self.rax_quality.set_title(
                "Transform quality: {:.2%}\nAverage displacement: {:.2f}".format(
                    perc_good, average_disp
                )
            )

            self.ax.set_title(
                "Reference vs Moving (Interactive), ref: {}, mov: {}".format(
                    n_ref, n_mov
                )
            )
            self.ax_reg.set_title(
                "Reference vs Registered" if not self.show_grid else "Warping Grid"
            )

            self.ax.set_axis_off()
            self.ax_reg.set_axis_off()

            # Prepare images
            img_ref = self.img_ref_.copy()
            img_ref[img_ref < self.th_ref] = 0

            img_mov = self.img_mov_.copy()
            img_mov[img_mov < self.th_movreg] = 0

            img_reg = self.img_reg.copy()
            img_reg[img_reg < self.th_movreg] = 0

            colored_grid = gray2rgb(self.grid_ / 255)
            colored_grid[self.df.jacobian <= 0, :] *= [0.9, 0, 0]
            warped_grid = self.df.warp(colored_grid)

            if self.ref_first:
                self.ax.imshow(img_ref, cmap=self.cmap_ref, alpha=self.alpha_ref)
                self.ax.imshow(img_mov, cmap=self.cmap_movreg, alpha=self.alpha_movreg)

                if self.show_grid:
                    self.ax_reg.imshow(warped_grid)

                else:
                    self.ax_reg.imshow(
                        img_ref, cmap=self.cmap_ref, alpha=self.alpha_ref
                    )
                    self.ax_reg.imshow(
                        img_reg, cmap=self.cmap_movreg, alpha=self.alpha_movreg
                    )

            else:
                self.ax.imshow(img_mov, cmap=self.cmap_movreg, alpha=self.alpha_movreg)
                self.ax.imshow(img_ref, cmap=self.cmap_ref, alpha=self.alpha_ref)

                if self.show_grid:
                    self.ax_reg.imshow(warped_grid)

                else:
                    self.ax_reg.imshow(
                        img_reg, cmap=self.cmap_movreg, alpha=self.alpha_movreg
                    )
                    self.ax_reg.imshow(
                        img_ref, cmap=self.cmap_ref, alpha=self.alpha_ref
                    )

            # Scatterplots
            refs_movs = [
                (k, v) for k, v in self.keypoints.items()
            ]  # THIS SETS THE ORDER

            refs_with_none = [x[0] for x in refs_movs]
            movs_with_none = [x[1] for x in refs_movs]

            colors_ref = [
                self.colors[x_ref] for x_ref, x_mov in refs_movs if x_ref is not None
            ]
            colors_mov = [
                self.colors[x_ref] for x_ref, x_mov in refs_movs if x_mov is not None
            ]

            # Reference points
            self.ax.scatter(
                [x[0] for x in refs_with_none if x is not None],
                [x[1] for x in refs_with_none if x is not None],
                marker=self.marker_ref,
                s=self.marker_size_ref,
                # label='ref',
                # c=[c for c in colors_ref if c is not None],
                # edgecolors='red',
                c=colors_ref,
            )

            # Moving points
            self.ax.scatter(
                [x[0] for x in movs_with_none if x is not None],
                [x[1] for x in movs_with_none if x is not None],
                marker=self.marker_mov,
                s=self.marker_size_mov,
                # label='mov',
                c=colors_mov,
            )

            ref_label = mlines.Line2D(
                [],
                [],
                color="black",
                marker=self.marker_ref,
                linestyle="None",
                markersize=self.marker_size_ref ** (1 / 2),
                label="ref",
            )
            mov_label = mlines.Line2D(
                [],
                [],
                color="black",
                marker=self.marker_mov,
                linestyle="None",
                markersize=self.marker_size_mov ** (1 / 2),
                label="mov",
            )

            if self.show_arrows:
                deltas = [
                    (m[0] - r[0], m[1] - r[1])
                    for r, m in refs_movs
                    if (r is not None and m is not None)
                ]
                x_del = [x[0] for x in deltas]
                y_del = [x[1] for x in deltas]

                x_pos = [
                    r[0] for r, m in refs_movs if (r is not None and m is not None)
                ]
                y_pos = [
                    r[1] for r, m in refs_movs if (r is not None and m is not None)
                ]

                for i in range(len(deltas)):
                    self.ax.arrow(
                        x_pos[i],
                        y_pos[i],
                        x_del[i],
                        y_del[i],
                        color=self.colors[(x_pos[i], y_pos[i])],
                    )

            self.ax.legend(handles=[ref_label, mov_label])

            plt.draw()

        def run(self):
            """Run the GUI."""

            def on_click(event):
                """Take action on a click.

                Parameters
                ----------
                event :  matplotlib.backend_bases.Event
                    In this case this is  matplotlib.backend_bases.LocationEvent.

                Notes
                -----
                We can use this to extract x and y coordinate of the click.

                """
                # Get coordinates of the click
                if event.inaxes != self.ax:  # only self.ax is an interactive axes
                    return

                x, y = int(event.xdata), int(event.ydata)
                print("x = %d, y = %d" % (x, y))

                # Clean axis (The logic is to draw everything from scratch)

                new_pair_mode = np.all(
                    [
                        x is not None
                        for x in (
                            self.keypoints.values()
                            if mode == "ref2mov"
                            else self.keypoints.keys()
                        )
                    ]
                )

                if new_pair_mode:
                    c = self.all_colors.popleft()
                    self.all_colors.append(c)

                    if mode == "ref2mov":
                        self.keypoints[(x, y)] = None
                        self.colors[(x, y)] = c
                    else:
                        self.keypoints[None] = (x, y)
                        self.colors[None] = c

                else:
                    if mode == "ref2mov":
                        self.keypoints[
                            [k for k, v in self.keypoints.items() if v is None][0]
                        ] = (x, y)
                    else:
                        self.keypoints[(x, y)] = self.keypoints[None]
                        del self.keypoints[None]
                        self.colors[(x, y)] = self.colors[None]
                        del self.colors[None]

                self._update_plots()

            def on_press(event):
                """Take action on a key press.

                Parameters
                ----------
                event :  matplotlib.backend_bases.Event
                    In this case this is  matplotlib.backend_bases.LocationEvent.

                """
                if event.key is None or ord(event.key) != 32:
                    return

                if event.inaxes != self.ax:
                    return

                x, y = int(event.xdata), int(event.ydata)

                # You can only remove reference points

                for diff_x in range(-self.epsilon, self.epsilon):
                    for diff_y in range(-self.epsilon, self.epsilon):
                        if (x + diff_x, y + diff_y) in self.keypoints:
                            del self.keypoints[(x + diff_x, y + diff_y)]
                            print("Deleted {}".format((x + diff_x, y + diff_y)))
                            self._update_plots()
                            return

            self.fig.canvas.mpl_connect("button_press_event", on_click)
            self.fig.canvas.mpl_connect("key_press_event", on_press)

            plt.show()

    helper_inst = HelperGlobal()
    helper_inst.run()

    return (
        helper_inst.df,
        helper_inst.keypoints,
        helper_inst.img_reg,
        helper_inst.interpolation_method,
        helper_inst.kernel,
    )  # noqa


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-f", "--fixed", help="Fixed image section (0-527)", type=int, default=300
    )
    parser.add_argument(
        "-m",
        "--moving",
        help="Moving image",
        type=str,
        default="examples/data/moving.png",
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path to saving folder",
        type=str,
        default="{}/.atlalign/label/".format(str(Path.home())),
    )
    parser.add_argument("-t", "--title", help="Title", type=str, default="")

    args = parser.parse_args()

    fixed = args.fixed
    moving = args.moving
    path = args.path
    title = args.title

    # Get fixed image
    img_ref = equalize_hist(nissl_volume()[fixed, ..., 0]).astype(np.float32)

    # Get moving image
    print(moving)
    img_mov_ = cv2.imread(moving, 0)
    img_mov = img_as_float32(cv2.imread(moving, 0))

    # Check folder exists
    path = path if path[-1] == "/" else path + "/"
    dir_exists = os.path.isdir(path)

    if dir_exists:
        warnings.warn(
            "Directory {} already exists, potentially rewriting data.".format(path)
        )
    else:
        os.makedirs(path)
        os.system("chmod " + path + " ug+rw")

    print(path)

    # Run GUI
    start_time = datetime.now()
    df, keypoints, img_reg, interpolation_method, kernel = run_GUI(
        img_ref,
        img_mov,
        mode="mov2ref",
        # mode='ref2mov'
        title=title,
    )

    time_spent = int((datetime.now() - start_time).total_seconds())

    # CHECKS
    assert img_reg.dtype == np.float32

    # STORE RESULTS
    # deltas
    delta_xy = np.zeros((320, 456, 2), dtype=np.float32)
    delta_xy[..., 0] = df.delta_x
    delta_xy[..., 1] = df.delta_y

    np.save("{}delta_xy.npy".format(path), delta_xy)

    # keypoints
    with open("{}keypoints.pkl".format(path), "wb") as f:
        pickle.dump(keypoints, f, pickle.HIGHEST_PROTOCOL)

    # registered
    plt.imsave("{}reg.png".format(path), img_reg)

    # metadata
    metadata = {
        "interpolation_method": interpolation_method,
        "kernel": kernel,
        "n_keypoints": len(keypoints),
        "sn": fixed,
        "time_spent": time_spent,
    }

    with open("{}metadata.json".format(path), "w") as out:
        json.dump(json.dumps(metadata), out)

    # Success txt
    with open("{}success.txt".format(path), "w"):
        pass
