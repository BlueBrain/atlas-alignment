"""Augmentation related tools."""

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

import imgaug.augmenters as iaa


def augmenter_1(p=0.99):
    """Create augmenter.

    Contains no coordinate transforms.

    Parameters
    ----------
    p : float
        Number in [0, 1] representing the probability of a random augmentation happening.

    Returns
    -------
    seq : iaa.Augmenter
        Augmenter where each augmentation was manually inspected and makes
        sense.

    """
    subsubseq_1 = iaa.Multiply(mul=(0.8, 1.2))
    subsubseq_2 = iaa.Sequential([iaa.Sharpen(alpha=(0, 1))])

    subsubseq_3 = iaa.Sequential([iaa.EdgeDetect(alpha=(0, 0.9))])

    subsubseq_4 = iaa.OneOf([iaa.GaussianBlur((0, 3.0)), iaa.AverageBlur(k=(2, 7))])

    subsubseq_5 = iaa.AdditiveGaussianNoise(loc=(0, 0.5), scale=(0, 0.2))

    subsubseq_6 = iaa.Add((-0.3, 0.3))

    subsubseq_7 = iaa.Invert(p=1)

    subsubseq_8 = iaa.CoarseDropout(p=0.25, size_percent=(0.005, 0.06))

    subsubseq_9 = iaa.SigmoidContrast(gain=(0.8, 1.2))

    subsubseq_10 = iaa.LinearContrast(alpha=(0.8, 1.2))

    subsubseq_11 = iaa.Sequential([iaa.Emboss(alpha=(0, 1))])

    seq = iaa.Sometimes(
        p,
        iaa.OneOf(
            [
                subsubseq_1,
                subsubseq_2,
                subsubseq_3,
                subsubseq_4,
                subsubseq_5,
                subsubseq_6,
                subsubseq_7,
                subsubseq_8,
                subsubseq_9,
                subsubseq_10,
                subsubseq_11,
            ]
        ),
    )

    return seq
