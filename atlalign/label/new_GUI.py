"""Graphical User Interface for manual registration."""
# The package atlalign is a tool for registration of 2D images.
#
# Copyright (C) 2021 EPFL/Blue Brain Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from collections import deque
from copy import deepcopy

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider
from skimage.color import gray2rgb

from atlalign.base import DisplacementField
from atlalign.visualization import create_grid


class HelperGlobal:
    """Just a way how to avoid using global variables.

    Parameters
    ----------
    img_ref : np.ndaray
        Reference image. Needs to be dtype == np.uint8.
    img_mov : np.ndarray
        Input image. Needs to be dtype == np.uint8 and the same shape as `img_ref`.
    mode : str, {'ref2mov', 'mov2ref'}
        If 'ref2mov' then the first point should be in the reference image and
        the other point in the moving one. For 'mov2ref' its vice versa.
    title : str,
        Additional title of the figure.

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

    def __init__(self, img_ref, img_mov, mode, title):
        self.mode = mode
        self.title = title

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
        self.th_movreg = 10 / 255

        self.ref_first = True
        self.show_grid = False
        self.show_arrows = True

        self.alpha_ref = 0.8
        self.alpha_movreg = 0.5
        self.alpha_movreg_prev = 0.0

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
        self.marker_ref = "."
        self.marker_mov = "+"
        self.marker_size_ref = 7 ** 2
        self.marker_size_mov = 7 ** 2

        self.all_colors = deque(
            ["red", "green", "blue", "yellow", "orange", "pink", "brown", "cyan"]
        )  # left start
        # self.all_colors = deque([cm.tab20(i / 100) for i in range(0, 100, 5)])

        # Attributes
        self.keypoints = {}  # # (x_ref, y_ref) -> (x_inp, y_inp)
        self.keypoints_prev = {}
        self.colors = {}  # (x_ref, y_ref) -> color ()

        self.epsilon = 3
        self.symmetric_registration = False

        # Modify keyboard shortcuts
        self.key_pan = "a"
        self.key_zoom_rect = "s"
        self.key_delete_ref_point = "d"
        self.key_reset_zoom = "f"
        self.key_swap_alpha = " "
        # Remove all default key bindings to avoid clashes
        for key, value in mpl.rcParams.items():
            if key.startswith("keymap."):
                value.clear()
        mpl.rcParams["keymap.pan"] = [self.key_pan]
        mpl.rcParams["keymap.zoom"] = [self.key_zoom_rect]
        mpl.rcParams["keymap.home"] = [self.key_reset_zoom]
        self.key_descriptions = {
            self.key_pan: "pan",
            self.key_zoom_rect: "zoom rectangle",
            self.key_reset_zoom: "reset zoom",
            self.key_delete_ref_point: "delete ref point",
            self.key_swap_alpha: "toggle alpha",
        }

        # Axis
        self.fig, (self.ax, self.ax_reg) = plt.subplots(1, 2, figsize=(20, 20))
        # self.fig.canvas.set_window_title(self.title)
        self.fig.tight_layout()
        self.fig.suptitle(self.title, fontsize=17)

        self.ax.set_axis_off()
        self.ax_reg.set_axis_off()

        self._define_widgets()

        # Initialize plots
        self._draw()

    def _make_buttons(self, y_pos):
        width, height = 0.15, 0.03

        # toggle reset
        self.reset_button = Button(
            plt.axes([0.01, y_pos, width, height]),
            "Reset",
            # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
        )

        def on_clicked(*args, **kwargs):
            self.keypoints = {}
            self._update_plots()

        self.reset_button.on_clicked(on_clicked)

        # Toggle symmetric registration
        self.toggle_symmetric_reg = Button(
            plt.axes([0.20, y_pos, width, height]),
            "",
            # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
        )

        def set_symmetric_reg_label():
            status_str = "[On]" if self.symmetric_registration else "[Off]"
            self.toggle_symmetric_reg.label.set_text(
                f"Symmetric Registration {status_str}"
            )

        def on_clicked(_event):
            self.symmetric_registration = not self.symmetric_registration
            set_symmetric_reg_label()
            # No no key points, so we have to force the redraw
            self._update_plots(force=True)

        set_symmetric_reg_label()
        self.toggle_symmetric_reg.on_clicked(on_clicked)

        # toggle ref_first
        self.ref_first_button = Button(
            plt.axes([0.4, y_pos, width, height]),
            "Change order",
            # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
        )

        def on_clicked(*args, **kwargs):
            self.ref_first = not self.ref_first
            self._update_plots()

        self.ref_first_button.on_clicked(on_clicked)

        # toggle show_arrows
        self.show_arrows_button = Button(
            plt.axes([0.6, y_pos, width, height]),
            "Show arrows",
            # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
        )

        def on_clicked(*args, **kwargs):
            self.show_arrows = not self.show_arrows
            self._update_plots()

        self.show_arrows_button.on_clicked(on_clicked)

        # toggle show_grid
        self.show_grid_button = Button(
            plt.axes([0.8, y_pos, width, height]),
            "Show grid",
            # color=[0.0, 1.0, 0.0] if self.ref_first else [1, 0, 0],
        )

        def on_clicked(*args, **kwargs):
            self.show_grid = not self.show_grid
            self._update_plots()

        self.show_grid_button.on_clicked(on_clicked)

    def _define_widgets(self):
        """Define all widgets."""
        self._make_buttons(y_pos=0.01)

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
            self.interpolation_method = self.interpolation_method_radio.value_selected
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
        self.rax_quality = plt.axes([0.45, 0.8, 0.10, 0.05], facecolor=axcolor)
        self.rax_quality.set_axis_off()

    def _update_plots(self, force=False):
        """Render the entire figure with the most recent parameters."""
        new_kps = self.keypoints != self.keypoints_prev
        new_ip = (
            self.interpolation_method != self.interpolation_method_prev
            or self.kernel != self.kernel_prev
        )
        is_complete = not np.any(
            [k is None or v is None for k, v in self.keypoints.items()]
        )

        if (new_kps and is_complete) or new_ip or force:
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

                # If symmetric registration, then add mirrored points
                if self.symmetric_registration:
                    _, width = self.img_ref_.shape[:2]
                    mirrored_kps = [
                        ((width - x1, y1), (width - x2, y2))
                        for (x1, y1), (x2, y2) in all_kps
                    ]
                    all_kps.extend(mirrored_kps)

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

        # Redraw figure while keeping the zoom/limits
        ax_xlim = self.ax.get_xlim()
        ax_ylim = self.ax.get_ylim()
        ax_reg_xlim = self.ax_reg.get_xlim()
        ax_reg_ylim = self.ax_reg.get_ylim()
        self._draw()
        self.ax.set(xlim=ax_xlim, ylim=ax_ylim)
        self.ax_reg.set(xlim=ax_reg_xlim, ylim=ax_reg_ylim)

    def _draw(self):
        self.ax.cla()
        self.ax_reg.cla()

        n_ref = len([x for x in self.keypoints.keys() if x is not None])
        n_mov = len([x for x in self.keypoints.values() if x is not None])

        perc_good = np.sum(self.df.jacobian > 0) / np.prod(self.df.shape)
        average_disp = self.df.average_displacement

        key_shortcuts = ", ".join(
            f"{description}: {key!r}"
            for key, description in self.key_descriptions.items()
        )
        self.rax_quality.set_title(
            f"Transform quality: {perc_good:.2%}\n"
            f"Average displacement: {average_disp:.2f}\n\n"
            f"(Key shortcuts: {key_shortcuts})"
        )

        self.ax.set_title(
            f"Reference vs Moving (Interactive), ref: {n_ref}, mov: {n_mov}"
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
                self.ax_reg.imshow(img_ref, cmap=self.cmap_ref, alpha=self.alpha_ref)
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
                self.ax_reg.imshow(img_ref, cmap=self.cmap_ref, alpha=self.alpha_ref)

        # Scatter plots
        refs_movs = [(k, v) for k, v in self.keypoints.items()]  # THIS SETS THE ORDER

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

            x_pos = [r[0] for r, m in refs_movs if (r is not None and m is not None)]
            y_pos = [r[1] for r, m in refs_movs if (r is not None and m is not None)]

            for i in range(len(deltas)):
                self.ax.arrow(
                    x_pos[i],
                    y_pos[i],
                    x_del[i],
                    y_del[i],
                    color=self.colors[(x_pos[i], y_pos[i])],
                )
        self.ax.legend(handles=[ref_label, mov_label])
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Take action on a click.

        Parameters
        ----------
        event :  matplotlib.backend_bases.LocationEvent
            The location event.

        Notes
        -----
        We can use this to extract x and y coordinate of the click.
        """
        # Can be [None, "ZOOM", "PAN"], don't handle clicks if
        # zooming or panning
        if self.ax.get_navigate_mode() is not None:
            return
        # Only self.ax are interactive axes
        if event.inaxes != self.ax:
            return

        # Get coordinates of the click
        x, y = int(event.xdata), int(event.ydata)

        # Clean axis (The logic is to draw everything from scratch)
        new_pair_mode = np.all(
            [
                x is not None
                for x in (
                    self.keypoints.values()
                    if self.mode == "ref2mov"
                    else self.keypoints.keys()
                )
            ]
        )

        if new_pair_mode:
            c = self.all_colors[0]
            self.all_colors.rotate()

            if self.mode == "ref2mov":
                self.keypoints[(x, y)] = None
                self.colors[(x, y)] = c
            else:
                self.keypoints[None] = (x, y)
                self.colors[None] = c

        else:
            if self.mode == "ref2mov":
                self.keypoints[
                    [k for k, v in self.keypoints.items() if v is None][0]
                ] = (x, y)
            else:
                self.keypoints[(x, y)] = self.keypoints[None]
                del self.keypoints[None]
                self.colors[(x, y)] = self.colors[None]
                del self.colors[None]

        self._update_plots()

    def on_press(self, event):
        """Take action on a key press.

        Parameters
        ----------
        event :  matplotlib.backend_bases.KeyEvent
            The key event fired.
        """
        if event.key is None:
            return

        key_pressed = event.key.lower()
        if key_pressed == self.key_delete_ref_point:
            print("Handling delete")
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
        elif key_pressed == self.key_swap_alpha:
            self.alpha_movreg, self.alpha_movreg_prev = (
                self.alpha_movreg_prev,
                self.alpha_movreg,
            )
            self.alpha_movreg_slider.set_val(self.alpha_movreg)

    def run(self):
        """Run the GUI."""
        # Register mouse and keyboard event callbacks
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)

        # Show the plot window
        plt.show()


def run_gui(img_ref, img_mov, mode="ref2mov", title=""):
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
    symmetric_registration : bool
        Whether or not the registration was symmetrized. If true then all the
        returned keypoints should be mirrored across a vertical line through
        the image. This can be done by setting x => (width - x) for all
        keypoints.
    img_reg : np.ndarray
        Registered image.
    interpolation_method : str
        Interpolation method
    kernel : str
        Kernel.
    """
    if not img_ref.shape == img_mov.shape:
        raise ValueError(
            "The fixed and moving image need to have the same shape. "
            f"{img_ref.shape} vs. {img_mov.shape}"
        )
    if not (img_ref.dtype == np.float32 and img_mov.dtype == np.float32):
        raise TypeError("Only works with float32 dtype")
    if mode not in {"ref2mov", "mov2ref"}:
        raise ValueError("The mode can only be ref2mov or mov2ref.")

    helper = HelperGlobal(img_ref, img_mov, mode, title)
    helper.run()

    return (
        helper.df,
        helper.keypoints,
        helper.symmetric_registration,
        helper.img_reg,
        helper.interpolation_method,
        helper.kernel,
    )  # noqa
