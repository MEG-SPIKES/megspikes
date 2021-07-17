# -*- coding: utf-8 -*-
# from typing import Union
# from pathlib import Path
import numpy as np
from scipy import signal, stats
from sklearn.cluster import KMeans
from sklearn import preprocessing

import xarray as xr

import mne

from alphacsc import GreedyCDL
from alphacsc.utils.signal import split_signal
from ..utils import create_epochs


class DecompositionICA():
    def __init__(self, n_components: int = 20):
        self.n_components = n_components

    def fit(self, X: xr.Dataset, y: mne.io.Raw):
        ica = mne.preprocessing.ICA(
            n_components=self.n_components, random_state=97)
        ica.fit(y)

        X['ica_components'] = (("ica_component", "channels"),
                               ica.get_components().T)
        # ICA timeseries [components x times]
        X['ica_sources'] = (("ica_component", "time"),
                            ica.get_sources(y).get_data())
        X['ica_components_kurtosis'] = (
            ("ica_component"),
            ica.score_sources(y, score_func=stats.kurtosis)
            )
        # ica.score_sources(data, score_func=stats.skew)
        return self

    def transform(self, X: xr.Dataset):
        return X
