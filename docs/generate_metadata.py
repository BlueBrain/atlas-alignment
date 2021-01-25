import pathlib
import datetime

import atlalign

metadata_template = \
    """---
packageurl: https://github.com/BlueBrain/atlas_alignment
major: {major_version}
description: Image registration with deep learning
repository: https://github.com/BlueBrain/atlas_alignment
externaldoc: https://bbpteam.epfl.ch/documentation/a.html#atlas-alignment
updated: {date}
maintainers: Jan Krepl
name: Atlas Alignment
license: BBP-internal-confidential
issuesurl: https://github.com/BlueBrain/atlas_alignment
version: {version}
contributors: Jan Krepl
minor: {minor_version}
---
"""

file_directory = pathlib.Path(__file__).parent.absolute()
metadata_path = file_directory / 'metadata.md'

version = atlalign.__version__
major_version = version.split('.')[0]
minor_version = version.split('.')[1]
date = datetime.datetime.now().strftime("%d/%m/%y")

metadata_instance = metadata_template.format(version=version,
                                             major_version=major_version,
                                             minor_version=minor_version,
                                             date=date)

with metadata_path.open('w') as f:
    f.write(metadata_instance)

print('Finished')
