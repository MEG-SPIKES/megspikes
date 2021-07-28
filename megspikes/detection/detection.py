import logging
from pathlib import Path
import warnings
from typing import Dict, Tuple, Union, Any

import mne
import numpy as np
import xarray as xr
from alphacsc import GreedyCDL
from alphacsc.utils.signal import split_signal
from scipy import signal, stats
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from ..utils import create_epochs

mne.set_log_level("ERROR")


class DecompositionICA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 20):
        self.n_components = n_components

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
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

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
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

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
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

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        logging.info("ICA peaks detection is done.")
        return X


class CleanDetections(TransformerMixin, BaseEstimator):
    """Select one spike in each diff_threshold ms using subcorr values

    Parameters
    ----------
    diff_threshold : int, optional
        refractory period between spikes in ms, by default 500
    n_spikes : int, optional
        select N spikes using subcorr value, by default 300
        This option is used to ensure that there are no more
        than a certain number of detections.
    """
    def __init__(self, diff_threshold: int = 500, n_cleaned_peaks: int = 300):
        self.diff_threshold = diff_threshold
        self.n_cleaned_peaks = n_cleaned_peaks

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):

        # mni_coord = X[0]["ica_peaks_localization"].values
        subcorr = X[0]["ica_peaks_subcorr"].values
        timestamps = X[0]["ica_peaks_timestamps"].values  # ms

        # all information about spikes in one array
        spikes = np.array([timestamps, subcorr]).T

        # sort spikes by time
        sorted_spikes_ind = np.argsort(spikes[:, 0])
        spikes = spikes[sorted_spikes_ind, :]
        cleaned_spikes = []

        for time in range(0, int(spikes[:, 0].max()), self.diff_threshold):
            # spikes in the diff window
            mask = (spikes[:, 0] > time) & (
                spikes[:, 0] <= (time + self.diff_threshold))
            spikes_in_range = spikes[mask, :]
            if len(spikes_in_range) > 0:
                # select max subcorr
                s_max_idx = np.argmax(spikes_in_range[:, 1])
                # append spike with max subcorr value
                cleaned_spikes.append(spikes_in_range[s_max_idx, :])

        cleaned_spikes = np.array(cleaned_spikes)

        # Select n_spikes with max subcorr
        sort_ind = np.argsort(
            cleaned_spikes[:, 1])[::-1][:self.n_cleaned_peaks]
        cleaned_spikes = cleaned_spikes[sort_ind, :]

        sorted_spikes_ind = np.argsort(cleaned_spikes[:, 0])
        cleaned_spikes = cleaned_spikes[sorted_spikes_ind, :]

        mask = np.isin(timestamps, cleaned_spikes[:, 0])
        X[0]["ica_peaks_selected"][:] = np.int32(mask)
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        logging.info("ICA peaks cleaning is done.")
        return X


class CropDataAroundPeaks(TransformerMixin, BaseEstimator):
    """Crop data around selected ICA peaks to run AlphaCSC

    Parameters
    ----------
    time : float, optional
        half of the window around the detection, by default 0.5
    """
    def __init__(self, time=0.5):
        self.time = time

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        all_peaks = X[0]["ica_peaks_timestamps"].values
        selected = np.array(X[0]["ica_peaks_selected"].values,
                            dtype=bool)
        timestamps = all_peaks[selected]

        # Add first sample
        timestamps += X[1].first_samp

        # Create epochs using timestamps
        epochs = create_epochs(
            X[1], timestamps, tmin=-self.time, tmax=self.time)

        # Epochs to raw
        epochs_data = epochs.copy().get_data()
        tr, ch, times = epochs_data.shape
        data = epochs_data.transpose(1, 0, 2).reshape(ch, tr*times)
        X = (X[0], mne.io.RawArray(data, epochs.info))
        del epochs, data
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.RawArray]:
        return X


class DecompositionAlphaCSC(TransformerMixin, BaseEstimator):
    def __init__(self, n_atoms: int = 3, atoms_width: float = 0.5,
                 sfreq: float = 200., greedy_cdl_kwarg: Dict = {
                     "rank1": True,
                     "uv_constraint": "separate",
                     "window": True,
                     "unbiased_z_hat": True,
                     "D_init": "chunk",
                     "lmbd_max": "scaled",
                     "reg": 0.1,
                     "n_iter": 100,
                     "eps": 1e-4,
                     "solver_z": "lgcd",
                     "solver_z_kwargs": {"tol": 1e-3, "max_iter": 100000},
                     "solver_d": "alternate_adaptive",
                     "solver_d_kwargs": {"max_iter": 300},
                     "sort_atoms": True,
                     "verbose": 0,
                     "random_state": 0},
                 split_signal_kwarg: Dict = {
                     "n_splits": 5,
                     "apply_window": True},
                 n_jobs: int = 1):
        self.n_atoms = n_atoms
        self.sfreq = sfreq
        self.atoms_width = atoms_width
        self.n_jobs = n_jobs
        self.greedy_cdl_kwarg = greedy_cdl_kwarg
        self.split_signal_kwarg = split_signal_kwarg
        self.n_times_atom = int(round(self.sfreq * self.atoms_width))

    def fit(self, X: Tuple[xr.Dataset, Union[mne.io.Raw, mne.io.RawArray]],
            y=None):
        data = X[1].get_data(picks='meg')
        data_split = split_signal(data, **self.split_signal_kwarg)
        cdl = GreedyCDL(n_atoms=self.n_atoms, n_times_atom=self.n_times_atom,
                        n_jobs=self.n_jobs, **self.greedy_cdl_kwarg)
        cdl.fit(data_split)
        z_hat = cdl.transform(data[None, :])[0]
        _, times = z_hat.shape
        _, ch = cdl.u_hat_.shape
        X[0]["alphacsc_z_hat"][:, :times] = z_hat
        X[0]["alphacsc_v_hat"][:, :] = cdl.v_hat_
        X[0]["alphacsc_u_hat"][:, :ch] = cdl.u_hat_

        del data, data_split, cdl
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Union[mne.io.Raw,
                                                      mne.io.RawArray]]:
        return X


class SelectAlphacscEvents():
    def __init__(self,
                 sensors: str = 'grad',
                 n_atoms: int = 3,
                 epoch_width_samples: int = 201,
                 sfreq: float = 200.,
                 z_hat_threshold: float = 3.,  # MAD
                 z_hat_threshold_min: float = 1.5,  # MAD
                 window_border: int = 10,  # samples
                 min_n_events: int = 15,
                 atoms_selection_gof: float = 80.,  # FIXME: 0.80
                 allow_lower_gof_grad: float = 20.,  # FIXME: 0.20
                 atoms_selection_n_events: int = 10,
                 cross_corr_threshold: float = 0.85,
                 atoms_selection_min_events: int = 0):
        """Select best events for the atom

        Parameters
        ----------
        epoch_width_samples : int, optional
            epochs width in samples, by default 201.
        sfreq : int, optional
            downsample freq, by default 200.
        z_hat_threshold : int, optional
            threshold for z-hat values in MAD, by default 3
        atoms_selection_gof : int, optional
            GOF value threshold, by default 80
        atoms_selection_n_events : int, optional
            n events threshold, by default 10
        cross_corr_threshold : float, optional
            cross-correlation on the max channel
            threshold, by default 0.85
        window_border : int, optional
            border of the window to search z-hat peaks,
            by default 10 samples (50 ms) FIXME: samples to ms
        """
        self.sensors = sensors
        self.n_atoms = n_atoms
        self.epoch_width_samples = epoch_width_samples
        self.sfreq = sfreq
        self.z_hat_threshold = z_hat_threshold
        self.z_hat_threshold_min = z_hat_threshold_min
        self.min_n_events = min_n_events
        self.atoms_selection_gof = atoms_selection_gof
        self.allow_lower_gof_grad = allow_lower_gof_grad
        self.atoms_selection_n_events = atoms_selection_n_events
        self.cross_corr_threshold = cross_corr_threshold
        self.window_border = window_border
        self.atoms_selection_min_events = atoms_selection_min_events

    def fit(self, X: Tuple[xr.Dataset, Union[mne.io.Raw, mne.io.RawArray]],
            y=None):
        n_channels = len(mne.pick_types(X[1].info, meg=True))
        self.v_hat = X[0]["alphacsc_v_hat"].values
        self.u_hat = X[0]["alphacsc_u_hat"][:, :n_channels].values
        self.z_hat = X[0]["alphacsc_z_hat"].values
        self.atoms_gof = X[0]["alphacsc_components_gof"].values
        ica_peaks = X[0]["ica_peaks_timestamps"].values
        ica_peaks_selected = np.array(
            X[0]["ica_peaks_selected"].values, dtype=bool)
        self.n_ica_peaks = sum(ica_peaks_selected)

        # Repeat for each atom
        for atom in range(self.n_atoms):
            n_events = 0
            z_hat_threshold = self.z_hat_threshold

            # decrease z-hat threshold until at least min_n_events events
            # selected
            while n_events < self.min_n_events:
                # select atom's best events according to z-hat
                (alignment, z_values, max_ch, selected_events, events) =\
                    self.find_max_z(atom, self.z_hat_threshold)

                n_events = np.sum(selected_events)

                # continue with lower z_hat_threshold if not enough events
                if n_events < self.min_n_events:
                    z_hat_threshold -= 0.2

                # stop if z-hat is too low
                if z_hat_threshold < self.z_hat_threshold_min:
                    break

            if n_events > self.atoms_selection_min_events:
                # make epochs only with best events for this atom
                epochs = create_epochs(
                   X[1], events[selected_events], tmin=-0.25, tmax=0.25)

                # Estimate goodness of the atom
                goodness = self.atom_goodness(
                   atom, selected_events, epochs, max_ch)

                # Align ICA peaks using z-hat
                mask = np.where(ica_peaks_selected)[0][selected_events]
                X[0]["alphacsc_detections_timestamps"][mask] = (
                    ica_peaks[mask] + alignment[selected_events])
                X[0]["alphacsc_detections_atom"][mask] = [atom]*len(mask)
                X[0]["alphacsc_detections_z_values"][
                    mask] = z_values[selected_events]
                X[0]["alphacsc_detections_goodness"][
                    mask] = [goodness]*len(mask)
        return self

    def find_max_z(self, atom: int, z_threshold: float):
        """find events with large z-peak before ICA peak

        One ICA peak:
        -0.5s---window_border---z_max---window_border---ICA peak--...--0.5s
        The same sketch in samples:
        0-10-----z_max-----90-100-----------------------201
        The same sketch in milliseconds:
        0-50-----z_max----450-500----------------------1001

        z_max - z_hat peak before ICA detections in the cropped raw data

        Parameters
        ----------
        atom : int
            AlphaCSC atom index
        z_threshold : float, optional
            threshold for z-hat values in MAD, by default 3
        Returns
        -------
        array_like, int [alignment]
            time to add to the ICA peaks (alignment to z_max);
        array_like, float [z_values]
            z_hat values in MAD;
        int [np.argmax(u_k)]
            max channel from u_hat
        array_like, bool [selected_events]
            events selected in the ICA peaks array
        array_like, np.int32 [events]
            timepoints of the selected events in the raw_cropped
        """

        window_border = int(self.window_border)
        half_event = self.epoch_width_samples // 2
        event_width = self.epoch_width_samples

        # timepoints of events in the raw cropped around ICA peaks
        events = [half_event + i*event_width for i in range(self.n_ica_peaks)]
        events = np.array(events)

        # load atoms properties
        z_k = self.z_hat[atom]
        u_k = self.u_hat[atom]
        v_k = self.v_hat[atom]

        # Reshape z_hat to (n_evets, event_width)
        z_events = np.array(
            [z_k[int(ev-half_event):int(ev+half_event)] for ev in events])

        alignment = np.zeros(self.n_ica_peaks)
        z_values = np.zeros(self.n_ica_peaks)
        selected = np.zeros(self.n_ica_peaks, dtype=np.bool)

        # estimate z-hat threshold
        z_mad = stats.median_absolute_deviation(z_k[z_k > 0])
        threshold = np.median(z_k[z_k > 0]) + z_mad*z_threshold

        for n, event in enumerate(z_events):
            # z_peak: max z-hat in the window
            z_in_window = event[window_border:half_event-window_border]
            z_peak = np.argmax(z_in_window)
            # value of the max z-hat

            z_max_value = event[z_peak + window_border]

            if z_max_value > threshold:
                # alignment of the ICA detections to the z-hat max peak
                # alignment to the center of the atom
                # max z-hat is at the beginning of v_hat but spike is later
                z_values[n] = z_max_value
                z_peak += window_border + len(v_k)//2
                alignment[n] = z_peak - half_event
                selected[n] = True

        # align events to the z-hat peak
        events += np.int32(alignment)
        return alignment, z_values, np.argmax(u_k), selected, events

    def atom_goodness(self, atom: int, selected_events: np.ndarray,
                      epochs: mne.Epochs, max_ch: int):
        """Evaluate atom quality

        Parameters
        ----------
        atom : int
            AlphaCSC atom index
        selected_events : array_like, bool
            bool array of the selected ICA peaks for this atom
        epochs : mne.Epochs
            epochs of the selected events
        max_ch : int
            max channel from the u_hat

        Returns
        -------
        float
            goodness value between 0 and 1
        """
        # First condition: absolute number of the events
        n_events = np.sum(selected_events)
        cond1 = self.atoms_selection_n_events - n_events
        if cond1 < 0:
            cond1 = 1
        else:
            cond1 = n_events / self.atoms_selection_n_events

        # Second condition: atoms localization GOF
        atoms_gof = self.atoms_gof[atom]
        if isinstance(self.sensors, str):
            # Allow lower GOF for grad
            if self.sensors == 'grad':
                atoms_gof -= self.allow_lower_gof_grad
        cond2 = self.atoms_selection_gof - atoms_gof
        if cond2 < 0:
            cond2 = 1
        else:
            cond2 = self.atoms_gof[atom] / self.atoms_selection_gof

        # Third condition: v_hat and max channel pattern similarity
        # cross-correlation epochs' max channel with atom's
        # shape inside core events
        max_ch_waveforms = epochs.get_data()[:, max_ch, :]
        cross_corr = np.zeros(max_ch_waveforms.shape[0])
        for i, spike in enumerate(max_ch_waveforms):
            a = spike/np.linalg.norm(spike)
            a = np.abs(a)
            v = self.v_hat[atom]/np.linalg.norm(self.v_hat[atom])
            v = np.abs(v)
            cross_corr[i] = np.correlate(a, v).mean()

        mean_cc = np.median(cross_corr)
        cond3 = self.cross_corr_threshold - mean_cc
        if cond3 < 0:
            cond3 = 1
        else:
            cond3 = mean_cc / self.cross_corr_threshold
        return (cond1 + cond2 + cond3) / 3

    def transform(self, X) -> Tuple[xr.Dataset, Union[mne.io.Raw,
                                                      mne.io.RawArray]]:
        return X


class ClustersMerging():
    """ Merge atoms from all runs into one atom's library:
        - Estimate atom's goodness threshold. Goodness represents
            cross-correlation of events inside the atom, number of
            the events and atom's u_hat gof
        - Estimate similarity between atoms using u_hat and v_hat.
            Merge atoms if similarity is higher 0.8.
        - Clean repetitions (detections in 10 ms window)
    """
    def __init__(self, dataset: Union[str, Path],
                 abs_goodness_threshold: float = 0.9,
                 max_corr: float = 0.8):
        self.dataset = dataset
        self.abs_goodness_threshold = abs_goodness_threshold
        self.max_corr = max_corr

    def fit(self, X: Any, y=None):
        X = xr.load_dataset(self.dataset)
        goodness_threshold = self._find_goodness_threshold(X)

        timestamps = X["alphacsc_detections_timestamps"].values
        atoms = X["alphacsc_detections_atom"].values
        goodness = X["alphacsc_detections_goodness"].values

        all_runs = X.run.values
        sensors = X.sensors.values

        atoms_lib_timestamps = []
        atoms_lib_atom = []
        atoms_lib_sensors = []
        atoms_lib_run = []
        atoms_lib_cluster_id = []

        goodness_threshold = 0.5

        u_hats = X["alphacsc_u_hat"].values
        v_hats = X["alphacsc_v_hat"].values

        cluster_id = 0
        for sens, sens_name in enumerate(sensors):
            n_channels = X["alphacsc_u_hat"].attrs[f"n_{sens_name}"]
            all_u = []
            all_v = []
            for run in all_runs:
                selected = timestamps[run, sens, :] != 0
                for atom in np.int32(np.unique(atoms[run, sens, selected])):
                    selected_atom = atoms[run, sens, selected] == atom
                    goodness_atom = goodness[run, sens, selected][
                        selected_atom]
                    assert len(np.unique(goodness_atom)) == 1
                    cond1 = goodness_atom[0] > goodness_threshold

                    u_atom = u_hats[run, sens, atom, :n_channels]
                    v_atom = v_hats[run, sens, atom, :]

                    cond2 = self._atoms_corrs_condition(
                        u_atom, v_atom, all_u, all_v)
                    if cond1 & cond2:
                        atoms_lib_timestamps += timestamps[
                            run, sens, selected][selected_atom].tolist()
                        atoms_lib_atom += [atom]*sum(selected_atom)
                        atoms_lib_sensors += [sens]*sum(selected_atom)
                        atoms_lib_run += [run]*sum(selected_atom)
                        atoms_lib_cluster_id += [
                            cluster_id]*sum(selected_atom)
                        all_u.append(u_atom)
                        all_v.append(v_atom)
                        cluster_id += 1

        assert len(atoms_lib_timestamps) > 0

        atoms_lib_timestamps = np.array(atoms_lib_timestamps)
        atoms_lib_atom = np.array(atoms_lib_atom)
        atoms_lib_sensors = np.array(atoms_lib_sensors)
        atoms_lib_run = np.array(atoms_lib_run)
        atoms_lib_cluster_id = np.array(atoms_lib_cluster_id)

        # clean repetitions and sort detections
        sort_unique_ind = np.unique(np.round(atoms_lib_timestamps/10),
                                    return_index=True)[1]
        atoms_lib_atom = atoms_lib_atom[sort_unique_ind]
        atoms_lib_sensors = atoms_lib_sensors[sort_unique_ind]
        atoms_lib_run = atoms_lib_run[sort_unique_ind]
        atoms_lib_timestamps = atoms_lib_timestamps[sort_unique_ind]
        atoms_lib_cluster_id = atoms_lib_cluster_id[sort_unique_ind]
        n_det = X["clusters_library_timestamps"].values.shape[0]

        if n_det > atoms_lib_atom.shape[0]:
            n_det = atoms_lib_atom.shape[0]

        X["clusters_library_timestamps"][:n_det] = atoms_lib_timestamps[:n_det]
        X["clusters_library_atom"][:n_det] = atoms_lib_atom[:n_det]
        X["clusters_library_sensors"][:n_det] = atoms_lib_sensors[:n_det]
        X["clusters_library_run"][:n_det] = atoms_lib_run[:n_det]
        # Unique ID for each cluster
        X["clusters_library_cluster_id"][:n_det] = atoms_lib_cluster_id[:n_det]
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        return (X, [])

    def _find_goodness_threshold(self, ds):
        goodness = ds["alphacsc_detections_goodness"].values.flatten()
        # Atoms' goodness
        # unique because goodness is repeated for each timestamp
        goodness = np.unique(goodness[goodness != 0])
        mean_goodness = np.mean(goodness) / len(goodness)
        std_goodness = np.std(goodness) / len(goodness)

        # FIXME: avoid situation when threshold is too high
        goodness_threshold = mean_goodness + std_goodness
        if goodness_threshold > self.abs_goodness_threshold:
            goodness_threshold = self.abs_goodness_threshold
        return goodness_threshold

    def _atoms_corrs_condition(self, u_atom, v_atom, all_u, all_v):
        # Avoid repetitions
        # TODO: here could be more intelligent merging step
        # IDEA: 'good' repetitions are admissible
        if (len(all_u) > 0) & (len(all_v) > 0):
            u_corrs = np.array(
                [np.correlate(u, u_atom) for u in np.vstack(all_u)])
            v_corrs = np.array(
                [np.correlate(v, v_atom) for v in np.vstack(all_v)])

            for u_c, v_c in zip(u_corrs, v_corrs):
                if (u_c > self.max_corr) & (v_c > self.max_corr):
                    return False
            return True
        else:
            return True
