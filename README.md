<p align="center">
  	<img alt="MEG-SPIKES " src="https://github.com/MEG-SPIKES/megspikes/blob/main/resources/logo.png" >

[![Python package](https://github.com/MEG-SPIKES/megspikes/actions/workflows/python-package.yml/badge.svg)](https://github.com/MEG-SPIKES/megspikes/actions/workflows/python-package.yml)
![Codecov](https://img.shields.io/codecov/c/github/MEG-SPIKES/megspikes?token=JPN3YML3LY)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CONDUCT.md)

## MEG-SPIKES

This repository contains functions for detecting, analyzing and evaluating epileptic spikes in MEG recording.

## Installation

The easiest way to install the package is using pip. You should clone the repository and install all dependencies:

```bash
git clone https://github.com/MEG-SPIKES/megspikes.git
cd megspikes/
pip install .

```

Examples and visualization require additional dependencies that could be installed using the following code:

```bash
pip install notebook hvplot pyqt5 pyvista pyvistaqt

```

## Examples

Examples of how to use this package are prepared in the [Jupyter Notebooks](examples/).

- [0_simulation.ipynb](examples/0_simulation.ipynb): simulation that was used to test this package and in the other examples.
- [1_manual_pipeline.ipynb](examples/1_manual_pipeline.ipynb): localization of the irritative area when spikes are already detected.
- [2_aspire_alphacsc_pipepline.ipynb](examples/2_aspire_alphacsc_pipepline.ipynb): full spikes detection pipeline and visualization of each step.

## Documentation

[aspire_alphacsc_default_params.yml](megspikes/aspire_alphacsc_default_params.yml) includes all default parameters that were used to run spike detection using combination of ASPIRE [[2]](#2) and AlphaCSC [[1]](#1).

[clusters_default_params.yml](megspikes/clusters_default_params.yml) describes all the parameters that were used for the irritative area prediction based on the detected events and their clustering.

## Contributing

All contributors are expected to follow the [code of conduct](CONDUCT.md).

## References

<a id="1">[1]</a>
La Tour, T. D., Moreau, T., Jas, M., & Gramfort, A. (2018). Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals. ArXiv:1805.09654 [Cs, Eess, Stat]. http://arxiv.org/abs/1805.09654

<a id="2">[2]</a>
Ossadtchi, A., Baillet, S., Mosher, J. C., Thyerlei, D., Sutherling, W., & Leahy, R. M. (2004). Automated interictal spike detection and source localization in magnetoencephalography using independent components analysis and spatio-temporal clustering. Clinical Neurophysiology, 115(3), 508–522. https://doi.org/10.1016/j.clinph.2003.10.036
