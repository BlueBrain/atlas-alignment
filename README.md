<img src="docs/source/logo/Atlas_Alignment_banner.jpg"/>

# Atlas Alignment

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://github.com/BlueBrain/atlas_alignment/releases">
    <img src="https://img.shields.io/github/v/release/BlueBrain/atlas_alignment" alt="Latest release" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/BlueBrain/atlas_alignment/blob/master/LICENSE.md">
    <img src="https://img.shields.io/github/license/BlueBrain/atlas_alignment" alt="License" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://travis-ci.com/BlueBrain/atlas_alignment">
    <img src="https://travis-ci.com/BlueBrain/atlas_alignment.svg?token=EpNKg1Tw8ZGyy3nCEcuz&branch=master" alt="Build status" />
    </a>
  </td>
</tr>
<tr>
	<td>Code Style</td>
	<td>
		<a href="https://github.com/psf/black">
		<img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
		</a>
		<a href="https://pycqa.github.io/isort/">
		<img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="Isort">
		</a>
		<a href="http://www.pydocstyle.org/">
		<img src="https://img.shields.io/badge/docstrings-pydocstyle-informational" alt="Pydocstyle">
		</a>
		<a href="https://flake8.pycqa.org/">
		<img src="https://img.shields.io/badge/PEP8-flake8-informational" alt="Pydocstyle">
		</a>
	</td>
</tr>
</table>

Atlas Alignment is a toolbox to perform multimodal image registration. It 
includes both traditional and supervised deep learning models. 

This project originated from the Blue Brain Project efforts on aligning mouse 
brain atlases obtained with ISH gene expression and Nissl stains. 


### Official documentation
All details related to installation and logic are described in the official documentation. Once
deployed a link will be added. Until then one can generate it locally following the instructions
below.


### Normal installation
Make sure the `pip` version is at least `19.1`.

```bash
pip install --upgrade pip
pip --version  # make sure >= 19.1
```

Then one can move on to installing the actual `atlalign` package:

```
pip install git+https://github.com/BlueBrain/atlas_alignment#egg=atlalign[tf]
```
The extras entry `[tf]` represents TensorFlow and can be dropped if it is already installed.


### Development installation + docs
As described above, make sure to get the most recent `pip`.
```
git clone https://github.com/BlueBrain/atlas_alignment
cd atlas_alignment
pip install -e .[dev,docs,tf]
```

Assuming the user ran the above it should be possible to generate the docs 
```
cd docs
make clean && make html
```
