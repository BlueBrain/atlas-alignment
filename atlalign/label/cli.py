"""Command line interface implementation."""
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
import argparse
import datetime
import pathlib
import sys
from contextlib import redirect_stdout


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
    parser.add_argument(
        "-g",
        "--force-grayscale",
        default=False,
        help="Force the images to be grayscale. Convert RGB to grayscale if necessary.",
        action="store_true",
    )
    args = parser.parse_args(argv)

    # Imports
    import matplotlib.pyplot as plt
    import numpy as np

    from atlalign.data import nissl_volume
    from atlalign.label.io import load_image
    from atlalign.label.new_GUI import run_gui

    # Read input images
    output_channels = 1 if args.force_grayscale else None
    if args.ref.isdigit():
        img_ref = nissl_volume()[int(args.ref), ..., 0]
    else:
        img_ref_path = pathlib.Path(args.ref)
        img_ref = load_image(
            img_ref_path,
            output_channels=output_channels,
            output_dtype="float32"
        )
    img_mov_path = pathlib.Path(args.mov)
    img_mov = load_image(
        img_mov_path,
        output_channels=output_channels,
        output_dtype="float32",
    )

    # Launch GUI
    (
        result_df,
        keypoints,
        symmetric_registration,
        img_reg,
        interpolation_method,
        kernel,
    ) = run_gui(img_ref, img_mov, mode="mov2ref" if args.swap else "ref2mov")

    # Dump results and metadata to disk
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    result_df.save(output_path / "df.npy")
    np.save(output_path / "img_reg.npy", img_reg)
    np.save(output_path / "img_ref.npy", img_ref)
    np.save(output_path / "img_mov.npy", img_mov)
    plt.imsave(output_path / "img_reg.png", img_reg)
    plt.imsave(output_path / "img_ref.png", img_ref)
    plt.imsave(output_path / "img_mov.png", img_mov)
    with open(output_path / "keypoints.csv", "w") as file, redirect_stdout(file):
        if args.swap:
            print("mov x,mov y,ref x,ref y")
        else:
            print("ref x,ref y,mov x,mov y")
        for (x1, y1), (x2, y2) in keypoints.items():
            print(f"{x1},{y1},{x2},{y2}")
    with open(output_path / "info.log", "w") as file, redirect_stdout(file):
        print("Timestamp :", datetime.datetime.now().ctime())
        print("")
        print("Parameters")
        print("----------")
        print("ref             :", args.ref)
        print("mov             :", args.mov)
        print("output_path     :", output_path.resolve())
        print("swap            :", args.swap)
        print("force-grayscale :", args.force_grayscale)
        print()
        print("Interpolation")
        print("-------------")
        print("Symmetric :", symmetric_registration)
        print("Method    :", interpolation_method)
        print("Kernel    :", kernel)
    print("Results were saved to", output_path.resolve())


if __name__ == "__main__":
    sys.exit(main())
