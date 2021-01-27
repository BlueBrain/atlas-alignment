# Atlas Alignment

### Overview
This project implements multiple alignment algorithms - including deep learning.

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
