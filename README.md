<p align="center">
<img alt="MEG-SPIKES " src="https://github.com/MEG-SPIKES/megspikes/raw/main/resources/logo.png" width="40%" height="auto"></p>

[![Python package](https://github.com/MEG-SPIKES/megspikes/actions/workflows/python-package.yml/badge.svg)](https://github.com/MEG-SPIKES/megspikes/actions/workflows/python-package.yml)
![Codecov](https://img.shields.io/codecov/c/github/MEG-SPIKES/megspikes?token=JPN3YML3LY)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CONDUCT.md)

## MEG-SPIKES

This repository contains functions for detecting, analyzing and evaluating epileptic spikes in MEG recording.

## Installation
**Optionally** create a fresh virtual environment:
```bash
conda create -n megspikes pip python=3.7
```

The easiest way to install the package is using pip:
```bash
pip install megspikes
```

To install the latest version of the package, you should clone the repository and install all dependencies:
```bash
git clone https://github.com/MEG-SPIKES/megspikes.git
cd megspikes/
pip install .
```

## Examples

Examples of how to use this package are prepared in the [Jupyter Notebooks](https://github.com/MEG-SPIKES/megspikes/blob/main/megspikes/examples/).

- [0_simulation.ipynb](https://github.com/MEG-SPIKES/megspikes/blob/main/megspikes/examples/0_simulation.ipynb): simulation used to test this package and in other examples.
- [1_manual_pipeline.ipynb](https://github.com/MEG-SPIKES/megspikes/blob/main/megspikes/examples/1_manual_pipeline.ipynb): localization of the irritative area for already detected (simulated) spikes.
- [2_aspire_alphacsc_pipepline.ipynb](https://github.com/MEG-SPIKES/megspikes/blob/main/megspikes/examples/2_aspire_alphacsc_pipepline.ipynb): full spikes detection pipeline and visualization of each step.

## Documentation

### ASPIRE AlphaCSC pipeline

Full detection pipeline is presented on the figure below. The image was created using [Scikit-learn](https://scikit-learn.org) __Pipeline__ module.
<p align="center">
<img alt="ASPIRE AlphaCSC pipeline" src="https://github.com/MEG-SPIKES/megspikes/raw/main/resources/aspire_alphacsc_pipeline.png"></p>

To reproduce this picture see [2_aspire_alphacsc_pipepline.ipynb](https://github.com/MEG-SPIKES/megspikes/blob/main/megspikes/examples/2_aspire_alphacsc_pipepline.ipynb).

As is it depicted on the figure, ASPIRE-AlphaCSC pipeline includes the following main steps:

1. ICA decomposition
   1. ICA components localization
   2. ICA components selection
   3. ICA peaks localization
   4. ICA peaks cleaning
2. AlphaCSC decomposition
   1. AlphaCSC atoms localization
   2. AlphaCSC events selection
   3. AlphaCSC atoms merging
      1. AlphaCSC atoms goodness evaluation
      2. AlphaCSC atoms selection

### Clusters localization and the irritative area prediction

Irritative zone prediction pipeline is presented on the figure below. The image was created using [Scikit-learn](https://scikit-learn.org) __Pipeline__ module.
<p align="center">
<img alt="ASPIRE AlphaCSC pipeline" src="https://github.com/MEG-SPIKES/megspikes/raw/main/resources/clusters_localization_pipeline.png" width="300px" height="auto"></p>

To reproduce this picture see [2_aspire_alphacsc_pipepline.ipynb](https://github.com/MEG-SPIKES/megspikes/blob/main/megspikes/examples/2_aspire_alphacsc_pipepline.ipynb) and [1_manual_pipeline.ipynb](https://github.com/MEG-SPIKES/megspikes/blob/main/megspikes/examples/1_manual_pipeline.ipynb).

### Parameters

[aspire_alphacsc_default_params.yml](https://github.com/MEG-SPIKES/megspikes/raw/main/megspikes/aspire_alphacsc_default_params.yml) includes all default parameters that were used to run spike detection using combination of ASPIRE [[2]](#2) and AlphaCSC [[1]](#1).

[clusters_default_params.yml](https://github.com/MEG-SPIKES/megspikes/raw/main/megspikes/clusters_default_params.yml) describes all the parameters that were used for the irritative area prediction based on the detected events and their clustering.

### Dependencies

#### Analysis

- [alphacsc](https://github.com/alphacsc/alphacsc)
- [mne](https://github.com/mne-tools/mne-python)
- [nibabel](https://github.com/nipy/nibabel)
- [numpy](https://github.com/numpy/numpy)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [scipy](https://github.com/scipy/scipy)

#### Data storing

- [pyyaml](https://github.com/yaml/pyyaml)
- [pandas](https://github.com/pandas-dev/pandas)
- openpyxl
- [xarray](https://github.com/pydata/xarray)
- netCDF4

#### Visualization

- [matplotlib](https://github.com/matplotlib/matplotlib)
- [nilearn](https://github.com/nilearn/nilearn)
- [panel](https://github.com/holoviz/panel)
- [param](https://github.com/holoviz/param)
- [notebook](https://github.com/jupyter/notebook)
- [hvplot](https://github.com/holoviz/hvplot)
- [pyqt5](https://www.riverbankcomputing.com/software/pyqt/)
- [pyvista](https://github.com/pyvista/pyvista)
- [pyvistaqt](https://github.com/pyvista/pyvistaqt)

#### Testing

- [pytest](https://github.com/pytest-dev/pytest)

## Contributing

All contributors are expected to follow the [code of conduct](https://github.com/MEG-SPIKES/megspikes/raw/main/CODE_OF_CONDUCT.md).

## References

<a id="1">[1]</a>
La Tour, T. D., Moreau, T., Jas, M., & Gramfort, A. (2018). Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals. ArXiv:1805.09654 [Cs, Eess, Stat]. http://arxiv.org/abs/1805.09654

<a id="2">[2]</a>
Ossadtchi, A., Baillet, S., Mosher, J. C., Thyerlei, D., Sutherling, W., & Leahy, R. M. (2004). Automated interictal spike detection and source localization in magnetoencephalography using independent components analysis and spatio-temporal clustering. Clinical Neurophysiology, 115(3), 508–522. https://doi.org/10.1016/j.clinph.2003.10.036
