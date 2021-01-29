"""Collection of modules related to using Allen Brain database and API."""

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

# functions
from atlalign.allen.sync import (  # noqa
    corners_rs9,
    download_dataset,
    get_reference_image,
    pir_to_xy_API,
    pir_to_xy_local,
    pir_to_xy_local_coronal,
    warp_rs9,
    xy_to_pir_API,
)

# variables
from atlalign.allen.utils import CACHE_FOLDER, get_image  # noqa
