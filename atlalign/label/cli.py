"""Command line interface implementation."""

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

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt


def main(argv=None):
    """Run CLI."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "ref",
        type=str,
        help="Either a path to a reference image or a number from [0, 528) "
        "representing the coronal dimension in the nissl stain volume.",
    )
    parser.add_argument(
        "mov",
        type=str,
        help="Path to a moving image. Needs to be of the same shape as " "reference.",
    )
    parser.add_argument(
        "output_path", type=str, help="Folder where the outputs will be stored."
    )
    parser.add_argument(
        "-s",
        "--swap",
        default=False,
        help="Swap to the moving to reference mode.",
        action="store_true",
    )

    args = parser.parse_args(argv)

    ref = args.ref
    mov = args.mov
    output_path = args.output_path
    swap = args.swap

    # Imports
    from atlalign.data import nissl_volume
    from atlalign.label import load_image, run_GUI

    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    if ref.isdigit():
        img_ref = nissl_volume()[int(ref), ..., 0]
    else:
        img_ref_path = pathlib.Path(ref)
        img_ref = load_image(img_ref_path)

    img_mov_path = pathlib.Path(mov)
    img_mov = load_image(
        img_mov_path, output_channels=1, keep_last=False, output_dtype="float32"
    )

    result_df = run_GUI(img_ref, img_mov, mode="mov2ref" if swap else "ref2mov")[0]

    img_reg = result_df.warp(img_mov)

    result_df.save(output_path / "df.npy")
    plt.imsave(str(output_path / "registered.png"), img_reg)


if __name__ == "__main__":
    sys.exit(main())
