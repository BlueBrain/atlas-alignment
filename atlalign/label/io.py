"""Input and output utilities for the command line interface."""

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

import cv2


def load_image(
    file_path,
    allowed_suffix=("jpg", "png"),
    output_dtype=None,
    output_channels=None,
    output_shape=None,
    input_shape=None,
    keep_last=False,
):
    """Load image.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to where the image stored.

    allowed_suffix : tuple
        List of allowed suffixes.

    output_dtype : str or None
        Determines the dtype of the output image. If None, then the same as input.

    output_channels : int, {1, 3} or None
        If 1 then grayscale, if 3 then RGB. If None then the sampe as the input image.

    output_shape : tuple
        Two element tuple representing (h_output, w_output).

    input_shape : tuple or None
        If None no assertion on the input shape. If not None then a tuple representing
        (h_input_expected, w_input_expected).

    keep_last : bool
        Only active if `output_channels=1`. If True, then the output has shape (h, w, 1). Else (h, w).

    Returns
    -------
    img : np.array
        Array of shape (h, w)

    """
    raw_input_ = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

    if input_shape is not None and input_shape != raw_input_[:2]:
        raise ValueError(
            "Asserted input shape {} different than actual one {}".format(
                input_shape, raw_input_.shape
            )
        )

    if output_channels is None or output_channels == 3:
        input_img = cv2.cvtColor(raw_input_, cv2.COLOR_BGR2RGB)

    elif output_channels == 1:
        input_img = cv2.cvtColor(raw_input_, cv2.COLOR_BGR2GRAY)

    else:
        raise ValueError("Invalid output channels: {}".format(output_channels))

    if output_shape is not None:
        input_img = cv2.resize(input_img, (output_shape[1], output_shape[0]))

    if input_img.ndim == 3 and input_img.shape[2] == 1 and not keep_last:
        input_img = input_img[..., 0]

    if output_dtype:
        if "float" in output_dtype:
            input_img = (input_img / 255).astype(output_dtype)

    return input_img
