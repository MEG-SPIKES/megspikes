from typing import Union, List, Tuple
import warnings
import logging

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

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None, **fit_params):
        ica = mne.preprocessing.ICA(
            n_components=self.n_components, random_state=97)
        ica.fit(X[1])
        components = ica.get_components().T
        (_, n_channels) = components.shape
        X[0]['ica_components'][:, :n_channels] = components
        # ICA timeseries [components x times]
        X[0]['ica_sources'][:, :] = ica.get_sources(X[1]).get_data()
        X[0]['ica_components_kurtosis'][:] = ica.score_sources(
            X[1], score_func=stats.kurtosis)
        # ica.score_sources(data, score_func=stats.skew)
        return self

    def transform(self, X, **transform_params) -> Tuple[xr.Dataset,
                                                        mne.io.Raw]:
        logging.info("ICA decomposition is done.")
        return X


class ComponentsSelection(TransformerMixin, BaseEstimator):
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
        self.gof = gof
        self.gof_abs = gof_abs
        self.kurtosis_min = kurtosis_min
        self.kurtosis_max = kurtosis_max
        self.n_components_if_nothing_else = n_components_if_nothing_else
        self.run = run
        self.n_runs = n_runs

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None, **fit_params):
        return self

    def transform(self, X, **transform_params) -> Tuple[xr.Dataset,
                                                        mne.io.Raw]:
        components = X[0]['ica_components'].values
        kurtosis = X[0]['ica_components_kurtosis'].values
        gof = X[0]['ica_components_gof'].values
        selected = X[0]['ica_components_selected'].values
        selected[:self.n_by_var] = 1  # first n components by variance
        selected[kurtosis < self.kurtosis_min] = 0
        selected[kurtosis > self.kurtosis_max] = 0
        selected[gof < self.gof] = 0

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

        X[0]['ica_components_selected'][:] = selected
        logging.info("ICA components selection is done.")
        return X


class PeakDetection(TransformerMixin, BaseEstimator):
    def __init__(self,
                 sign: int = -1,
                 sfreq: int = 200,
                 h_filter: float = 20.,
                 l_filter: float = 90,
                 filter_order: int = 3,
                 prominence: float = 7.,
                 wlen: float = 2000.,
                 rel_height: float = 0.5,
                 width: float = 10.,
                 n_detections_threshold: int = 2000) -> None:
        # FIXME: sign should be list
        self.sign = sign
        self.sfreq = sfreq
        self.h_filter = h_filter
        self.l_filter = l_filter
        self.filter_order = filter_order
        self.prominence = prominence
        self.wlen = wlen
        self.rel_height = rel_height
        self.width = width
        self.n_detections_threshold = n_detections_threshold

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None, **fit_params):
        sources = X[0]["ica_sources"].values
        selected = X[0]["ica_components_selected"].values

        source_ind = np.where(selected.flatten() > 0)[0].tolist()

        timestamps = np.array([])
        channels = np.array([])

        # Loop to find required amount of the detections
        n_detections = 0
        while n_detections < self.n_detections_threshold:
            for ind in source_ind:
                data = sources[ind, :]
                for s in [self.sign]:
                    # Detect peaks
                    if s == -1:
                        data *= -1
                    peaks, _ = self._find_peaks(data)
                    timestamps = np.append(timestamps, peaks)
                    channels = np.append(channels, np.ones_like(peaks)*ind)

            n_detections = len(timestamps)

            # prominence threshold goes down
            if n_detections < self.n_detections_threshold:
                self.prominence -= 0.5
                if self.prominence < 2:
                    n_detections = self.n_detections_threshold
                else:
                    timestamps = np.array([])
                    channels = np.array([])
        if len(timestamps) > self.n_detections_threshold:
            n_det = self.n_detections_threshold
        else:
            n_det = len(timestamps)
        X[0]["ica_peaks_timestamps"][:n_det] = timestamps[:n_det]
        if n_det == 0:
            warnings.warn("NO ICA peaks!!!")
        return self

    def _find_peaks(self, data):
        freq = np.array([self.h_filter, self.l_filter]) / (self.sfreq / 2.0)
        b, a = signal.butter(self.filter_order, freq, "pass")
        data = signal.filtfilt(b, a, data)

        # robust_scaleprominence {prominence}
        data = preprocessing.robust_scale(data)

        peaks, props = signal.find_peaks(
            data, prominence=self.prominence, wlen=self.wlen,
            rel_height=self.rel_height, width=self.width)
        # delete peaks at the beginig and at the end of the recording
        window = self.sfreq
        peaks = peaks[(peaks > window) & (peaks < len(data)-window)]
        return peaks, props

    def transform(self, X, **transform_params) -> Tuple[xr.Dataset,
                                                        mne.io.Raw]:
        logging.info("ICA peaks detection is done.")
        return X
