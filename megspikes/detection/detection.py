from typing import Union, List, Tuple
import numpy as np
from scipy import signal, stats
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

import xarray as xr

import mne
mne.set_log_level("ERROR")

from alphacsc import GreedyCDL
from alphacsc.utils.signal import split_signal
from ..utils import create_epochs


class DecompositionICA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 20):
        self.n_components = n_components

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        ica = mne.preprocessing.ICA(
            n_components=self.n_components, random_state=97)
        ica.fit(X[1])

        X[0]['ica_components'] = (("ica_component", "channels"),
                                  ica.get_components().T)
        # ICA timeseries [components x times]
        X[0]['ica_sources'] = (("ica_component", "time"),
                               ica.get_sources(X[1]).get_data())
        X[0]['ica_components_kurtosis'] = (
            ("ica_component"),
            ica.score_sources(X[1], score_func=stats.kurtosis)
            )
        # ica.score_sources(data, score_func=stats.skew)
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        return X


class ComponentsSelection():
    """Select ICA components for analysis

    Parameters
    ----------
    n_by_var : int, optional
        N first components selected by variance, by default 10
    gof : float, optional
        components dipole fitting threshold, by default 80.
    gof_abs : float, optional
        absolute dippole fitting threshold, by default 95.
    kurtosis_min : float, optional
        minimal kurtosis score of the ICA component, by default 1.
    kurtosis_max : float, optional
        maximal kurtosis score of the ICA component, by default 10.
    run : int, optional
        run number, by default 0
    n_runs : int, optional
        all runs in the analysis, by default 4
    """

    def __init__(self,
                 n_by_var: int = 10,
                 gof: float = 80.,
                 gof_abs: float = 95.,
                 kurtosis_min: float = 1.,
                 kurtosis_max: float = 10.,
                 n_runs: int = 4,
                 n_components_if_nothing_else: int = 7,
                 run: int = 0) -> None:

        self.n_by_var = n_by_var  # n components selected by variance
        self.gof_param = gof
        self.gof_abs = gof_abs
        self.kurtosis_min = kurtosis_min
        self.kurtosis_max = kurtosis_max
        self.n_components_if_nothing_else = n_components_if_nothing_else
        self.run = run
        self.n_runs = n_runs

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        components = X[0]['ica_components'].values
        kurtosis = X[0]['ica_components_kurtosis'].values
        gof = X[0]['ica_components_gof'].values
        selected = X[0]['ica_components_selected'].values
        selected[:self.n_by_var] = 1  # first n components by variance
        selected[kurtosis < self.kurtosis_min] = 0
        selected[kurtosis > self.kurtosis_max] = 0
        selected[gof < self.gof_param] = 0

        # if component has gof > 97 include it in any case,
        # ignoring the other parameters
        selected[gof > self.gof_abs] = 1

        # select at 7 components by gof if nothing else was selected
        if np.sum(selected) == 0:
            selected[
                np.argsort(gof)[::-1][:self.n_components_if_nothing_else]] = 1

        if self.run != 0:
            # cluster components
            # IDEA: cluster coordinates not components
            n_runs = self.n_runs - 1
            if np.sum(selected) < n_runs:
                selected[np.argsort(gof)[::-1][:n_runs]] = 1

            # NOTE: n_clusters should be equal n_runs
            kmeans = KMeans(n_clusters=n_runs, random_state=0).fit(
                components[selected == 1])
            labels = kmeans.labels_

            # Select only one cluster for each run
            new_sel = selected[selected == 1]
            new_sel[labels + 1 != self.run] = 0
            selected[selected == 1] = new_sel

        X[0]['ica_components_selected'].values = selected
        return X
