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
from ..database.database import (check_and_read_from_dataset,
                                 check_and_write_to_dataset)

mne.set_log_level("ERROR")


class DecompositionICA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 20):
        self.n_components = n_components

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        self.ica = mne.preprocessing.ICA(
            n_components=self.n_components, random_state=97)
        self.ica.fit(X[1])
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        assert X[0].time.attrs['sfreq'] == X[1].info['sfreq'], (
            "Wrong sfreq of the fif file or database time coordinate")
        # ica components
        check_and_write_to_dataset(
            X[0], 'ica_components', self.ica.get_components().T)
        # ICA timeseries, shape=(components, times)
        check_and_write_to_dataset(
            X[0], 'ica_sources', self.ica.get_sources(X[1]).get_data())
        # compute kurtosis for each ica component
        check_and_write_to_dataset(
            X[0], 'ica_component_properties',
            self.ica.score_sources(X[1], score_func=stats.kurtosis),
            dict(ica_component_property='kurtosis'))
        # ica.score_sources(data, score_func=stats.skew)
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
        run number. NOTE: starting from 0, by default 0
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
        # load data from database
        components = check_and_read_from_dataset(
            X[0], 'ica_components')
        kurtosis = check_and_read_from_dataset(
            X[0], 'ica_component_properties',
            dict(ica_component_property='kurtosis'))
        gof = check_and_read_from_dataset(
            X[0], 'ica_component_properties',
            dict(ica_component_property='gof'))

        # Run the ICA components selection
        selected = self.select_ica_components(
            components, kurtosis, gof)

        # Write the selected components to the dataset
        check_and_write_to_dataset(
            X[0], 'ica_component_selection', selected)
        logging.info("ICA components selection is done.")
        return X

    def select_ica_components(self, components: np.ndarray,
                              kurtosis: np.ndarray,
                              gof: np.ndarray) -> np.ndarray:
        """Select ICA components to run pipeline.

        Parameters
        ----------
        components : np.ndarray
            shape channels by the number of components
        kurtosis : np.ndarray
            one value for each ICA component
        gof : np.ndarray
            one value between 0 and 1 for each ICA component

        Returns
        -------
        np.ndarray
            one value (o or 1) for each ICA component. 1 means that component
            selected.
        """
        selected = np.ones_like(kurtosis)
        selected[:self.n_by_var] = 1  # first n components by variance
        selected[kurtosis < self.kurtosis_min] = 0
        selected[kurtosis > self.kurtosis_max] = 0
        selected[gof < self.gof] = 0

        # if component has gof > 97 include it in any case,
        # ignoring the other parameters
        selected[gof > self.gof_abs] = 1

        # select at least 7 components by gof if nothing else was selected
        if np.sum(selected) == 0:
            selected[np.argsort(gof)[::-1][
                :self.n_components_if_nothing_else]] = 1

        # if only one run or this is the first run
        if self.run != 0:
            # cluster components
            # IDEA: cluster ICA components localization coordinates but
            # not components
            n_runs = self.n_runs - 1
            if np.sum(selected) < n_runs:
                selected[np.argsort(gof)[::-1][:n_runs]] = 1

            # NOTE: n_clusters should at least equal n_runs
            kmeans = KMeans(n_clusters=n_runs, random_state=0).fit(
                components[selected == 1])
            labels = kmeans.labels_

            if len(np.unique(labels)) == n_runs:
                # Select only one cluster for each run
                new_sel = selected[selected == 1]
                new_sel[labels + 1 != self.run] = 0
                selected[selected == 1] = new_sel
            else:
                warnings.warn("Not enough ICA components for clustering")
                # select one component by GOF
                selected[np.argsort(gof)[::-1][:n_runs]] = 0
                selected[np.argsort(gof)[::-1][n_runs]] = 1

        if sum(selected) == 0:
            warnings.warn(
                f"""Can't select ICA components, select first
                {self.n_components_if_nothing_else} by GOF""")
            selected[np.argsort(gof)[::-1][
                :self.n_components_if_nothing_else]] = 1

        return selected


class PeakDetection(TransformerMixin, BaseEstimator):
    """Detect peaks on the ICA components using scipy.signal.find_peaks method.

    Parameters
    ----------
    sign : int, optional
        Detect positive or negative peaks. Multiply the data by sign value,
        by default -1
    sfreq : int, optional
        sample frequency of the ICA sources, by default 200
    h_filter : float, optional
        heighpass filter, by default 20.
    l_filter : float, optional
        lowpass filter, by default 90
    filter_order : int, optional
        Filter order, by default 3
    prominence : float, optional
        Amplitude of the peaks to detect. Prominence is decreased automatically
        if the number of detection is less than n_detections_threshold,
        by default 7.
    wlen : float, optional
        Refractory window around the detected peak, by default 2000.
    rel_height : float, optional
        [description], by default 0.5
    width : float, optional
        Width of the peak in samples, by default 10.
    n_detections_threshold : int, optional
        Minimal number of the detections, by default 2000
    """
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
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        sources = check_and_read_from_dataset(
            X[0], 'ica_sources')
        selected = check_and_read_from_dataset(
            X[0], 'ica_component_selection')

        detections = np.zeros_like(X[0].time.values)
        ica_index = np.zeros_like(X[0].time.values)
        # find timestamps and corresponding sources
        det_ind, ica_ind = self.find_ica_peaks(sources, selected)
        detections[det_ind] = 1
        ica_index[det_ind] = ica_ind

        check_and_write_to_dataset(
            X[0], 'detection_properties',
            detections,
            dict(detection_property='detection'))
        check_and_write_to_dataset(
            X[0], 'detection_properties',
            ica_index,
            dict(detection_property='ica_component'))

        X[0]['detection_properties'].attrs["ica_components_filtering"] = (
            self.h_filter, self.l_filter)
        logging.info("ICA peaks detection is done.")
        return X

    def find_ica_peaks(self, sources: np.ndarray,
                       selected: np.ndarray) -> np.ndarray:
        """Find peaks in ICA sources.

        Parameters
        ----------
        sources : np.ndarray
            shape: n_ica_components by length of MEG recording
        selected : np.ndarray
            1d array with values 1 and 0; 1 - selected ICA component

        Returns
        -------
        np.ndarray
            detected peaks index
        np.ndarray
            source index for each peak
        """
        source_ind = np.where(selected.flatten() > 0)[0]

        timestamps = np.array([], dtype=np.int64)
        channels = np.array([], dtype=np.int64)

        # Loop to find required amount of the detections
        n_detections = 0
        while n_detections < self.n_detections_threshold:
            for ind in source_ind:
                data = sources[ind, :]
                for s in [self.sign]:
                    # Detect only negative peaks
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
                    timestamps = np.array([], dtype=np.int64)
                    channels = np.array([], dtype=np.int64)
        sort_unique_ind = np.unique(timestamps, return_index=True)[1]
        return timestamps[sort_unique_ind], channels[sort_unique_ind]

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


class CleanDetections(TransformerMixin, BaseEstimator):
    """Select one spike in each diff_threshold ms using subcorr values

    Parameters
    ----------
    diff_threshold : float, optional
        refractory period between spikes in s, by default 0.5
    n_spikes : int, optional
        select N spikes using subcorr value, by default 300
        This option is used to ensure that there are no more
        than a certain number of detections.
    """
    def __init__(self, diff_threshold: float = 0.5,
                 n_cleaned_peaks: int = 300) -> None:
        self.diff_threshold = diff_threshold
        self.n_cleaned_peaks = n_cleaned_peaks

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        detection = check_and_read_from_dataset(
            X[0], 'detection_properties',
            dict(detection_property='detection'))
        detection_mask = detection > 0
        # samples
        timestamps = np.where(detection_mask)[0]

        subcorr = check_and_read_from_dataset(
            X[0], 'detection_properties',
            dict(detection_property='subcorr'))
        subcorrs = subcorr[detection_mask]

        sfreq = X[0].time.attrs['sfreq']
        selected_peaks = self.clean_detections(timestamps, subcorrs, sfreq)
        selection = np.zeros_like(detection)
        selection[np.where(detection_mask)[0][selected_peaks == 1]] = 1
        check_and_write_to_dataset(
            X[0], 'detection_properties', selection,
            dict(detection_property='selected_for_alphacsc'))
        logging.info("ICA peaks cleaning is done.")
        return X

    def clean_detections(self, timestamps: np.ndarray, subcorr: np.ndarray,
                         sfreq: float = 200.) -> np.ndarray:
        selection = np.zeros_like(timestamps)
        window = int(self.diff_threshold*sfreq)
        for time in range(0, int(timestamps.max()), window):
            # timestamp in the diff window
            mask = (timestamps > time) & (timestamps <= (time + window))
            if len(timestamps[mask]) > 0:
                # select max subcorr
                subcorr_max_idx = np.argmax(subcorr[mask])
                # add timepoint with max subcorr value to selection
                selection[np.where(mask)[0][subcorr_max_idx]] = 1

        # Select n_spikes with max subcorr
        not_selected = np.argsort(
            subcorr[selection == 1])[::-1][self.n_cleaned_peaks:]
        selection[not_selected] = 0
        return selection


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

    def fit(self, X: Tuple[xr.Dataset, Union[mne.io.Raw]], y=None):
        # Prepare subset of full MEG data for fitting
        cropped_raw = self._crop_raw_around_peaks(X[0], X[1])

        # fit AlphaCSC
        data = cropped_raw.get_data(picks='meg')
        data_split = split_signal(data, **self.split_signal_kwarg)
        self.cdl = GreedyCDL(
            n_atoms=self.n_atoms, n_times_atom=self.n_times_atom,
            n_jobs=self.n_jobs, **self.greedy_cdl_kwarg)
        self.cdl.fit(data_split)
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Union[mne.io.Raw]]:
        # Transform the full MEG file
        data = X[1].get_data(picks='meg')
        z_hat = self.cdl.transform(data[None, :])[0]
        # align z-hat with MEG recording
        full_time = data.shape[1]
        atoms, times = z_hat.shape
        _z_hat = np.zeros((atoms, full_time))
        _z_hat[:, full_time-times:] = z_hat

        # Save results to dataset
        check_and_write_to_dataset(X[0], 'alphacsc_z_hat', _z_hat)
        check_and_write_to_dataset(X[0], 'alphacsc_v_hat', self.cdl.v_hat_)
        check_and_write_to_dataset(X[0], 'alphacsc_u_hat', self.cdl.u_hat_)
        del data
        logging.info("AlphaCSC decomposition is done.")
        return X

    def _crop_raw_around_peaks(self, ds: xr.Dataset, raw: mne.io.Raw):
        """Crop data around selected ICA peaks to run AlphaCSC

        Parameters
        ----------
        time : float, optional
            half of the window around the detection, by default 0.5
        """
        assert ds.time.attrs['sfreq'] == raw.info['sfreq'], (
            "Wrong sfreq of the fif file or database time coordinate")

        detections = check_and_read_from_dataset(
            ds, 'detection_properties',
            dict(detection_property='selected_for_alphacsc'))
        detection_mask = detections > 0
        timestamps = np.where(detection_mask)[0]

        # Add first sample
        timestamps += raw.first_samp
        # Create epochs using timestamps
        epochs = create_epochs(
            raw, timestamps, tmin=-self.atoms_width, tmax=self.atoms_width)
        # Epochs to raw
        epochs_data = epochs.copy().get_data()
        tr, ch, times = epochs_data.shape
        assert len(timestamps) == tr, (
            "Number of cropped epochs is not equal number of the detected ICA"
            "peaks.")
        data = epochs_data.transpose(1, 0, 2).reshape(ch, tr*times)
        return mne.io.RawArray(data, epochs.info)


class SelectAlphacscEvents():
    """Select best events for the atom.

    Parameters
    ----------
    epoch_width_samples : int, optional
        epochs of events in cropped data width in samples, by default 201.
    atom_width : float, optional
        atom width in seconds, by default 0.5
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
    def __init__(self,
                 sensors: str = 'grad',
                 n_atoms: int = 3,
                 epoch_width_samples: int = 201,
                 atom_width: float = 0.5,
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
        self.sensors = sensors
        self.n_atoms = n_atoms
        self.epoch_width_samples = epoch_width_samples
        self.atom_width = atom_width
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
        self.atom_width_samples = int(atom_width * sfreq)

    def fit(self, X: Tuple[xr.Dataset, Union[mne.io.Raw]], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Union[mne.io.Raw]]:
        assert X[0].time.attrs['sfreq'] == X[1].info['sfreq'], (
            "Wrong sfreq of the fif file or database time coordinate")

        detections = check_and_read_from_dataset(
            X[1], 'detection_properties',
            dict(detection_property='selected_for_alphacsc'))
        detection_mask = detections > 0
        ica_peaks = np.where(detection_mask)[0]

        v_hat = check_and_read_from_dataset(X[0], 'alphacsc_v_hat')
        u_hat = check_and_read_from_dataset(X[0], 'alphacsc_u_hat')
        z_hat = check_and_read_from_dataset(X[0], 'alphacsc_z_hat')
        gof = check_and_read_from_dataset(
            X[0], 'alphacsc_atoms_properties',
            dict(ica_component_property='gof'))

        # Select alphaCSC peaks
        (alphacsc_detection, alphacsc_atom, alphacsc_ica_alignment
         ) = self.select_alphacsc_peaks(z_hat, ica_peaks)

        # Write results to the dataset
        check_and_write_to_dataset(
            X[0], 'adetection_properties', alphacsc_detection,
            dict(ica_component_property='alphacsc_detection'))
        check_and_write_to_dataset(
            X[0], 'detection_properties', alphacsc_atom,
            dict(ica_component_property='alphacsc_atom'))
        check_and_write_to_dataset(
            X[0], 'detection_properties', alphacsc_ica_alignment,
            dict(ica_component_property='alphacsc_ica_alignment'))

        # Evaluate atoms' goodness
        goodness = self.evaluate_atoms_goodness(
            X[1], u_hat, v_hat, alphacsc_detection, alphacsc_atom, gof)

        # Write results to the dataset
        check_and_write_to_dataset(
            X[0], 'alphacsc_atoms_properties', goodness,
            dict(ica_component_property='goodness'))

        logging.info("AlphaCSC events are selected.")
        return X

    def select_alphacsc_peaks(self, z_hat: np.ndarray, ica_peaks: np.ndarray
                              ) -> Tuple[np.ndarray]:
        """Select z-hat peaks before the ICA detections.

        Parameters
        ----------
        z_hat : np.ndarray
            z-hat timeseries. Shape: atoms by times (length of the MEG data)
        ica_peaks : np.ndarray
            1D binary array of the ICA peaks (length of the MEG data)

        Returns
        -------
        alphacsc_detection
            1D binary array of the alphaCSC detection (length of the MEG data)
        alphacsc_atom
            array of the alphaCSC atoms (length of the MEG data)
        ica_alphacsc_aligned
            array with ica peak index for each AlphaCSC detection
            (same length as MEG recording)
        """
        alphacsc_detection = np.zeros_like(ica_peaks)
        alphacsc_atom = np.zeros_like(ica_peaks)
        alphacsc_ica_alignment = np.zeros_like(ica_peaks)

        # Repeat for each atom
        for atom in range(self.n_atoms):

            n_events = 0
            z_hat_threshold = self.z_hat_threshold

            # decrease z-hat threshold until at least min_n_events events
            # selected
            while n_events < self.min_n_events:
                # select atom's best events according to z-hat
                detections, alignments = self._find_max_z(
                    z_hat, ica_peaks, z_hat_threshold)

                n_events = np.sum(detections)

                # continue with lower z_hat_threshold if not enough events
                if n_events < self.min_n_events:
                    z_hat_threshold -= 0.2

                # stop if z-hat is too low
                if z_hat_threshold < self.z_hat_threshold_min:
                    break

            if n_events > self.atoms_selection_min_events:
                detection_mask = detections > 0
                alphacsc_detection[detection_mask] = 1
                alphacsc_atom[detection_mask] = atom
                alphacsc_ica_alignment[detection_mask] = alignments[
                    detection_mask]
        return alphacsc_detection, alphacsc_atom, alphacsc_ica_alignment

    def _find_max_z(self, z_hat: np.ndarray, ica_peaks: np.ndarray,
                    z_threshold: float) -> Tuple[np.ndarray]:
        """Find events with large z-peak before ICA peak.

        One ICA peak:
        -0.5s---window_border---z_max---window_border---ICA peak--...--0.5s

        The same sketch in samples:
        0-10-----z_max-----90-100-----------------------201

        The same sketch in milliseconds:
        0-50-----z_max----450-500----------------------1001

        z_max - z_hat peak before ICA detections in the cropped raw data

        Parameters
        ----------
        z_hat : np.ndarray
            z-hat timeseries with the same length as MEG recording
        ica_peaks : np.ndarray
            binary array with the same length as MEG recording
        z_threshold : float
            threshold for z-hat values in MAD, by default 3

        Returns
        -------
        alphacsc_detection
            binary array with the same length as MEG recording
        ica_alphacsc_aligned
            array with ica peak index for each AlphaCSC detection
            (same length as MEG recording)
        """

        border = int(self.window_border)
        half_event = self.epoch_width_samples // 2
        half_atom_width = self.atom_width_samples // 2
        # event_width = self.epoch_width_samples

        # estimate z-hat threshold
        z_mad = stats.median_abs_deviation(z_hat[z_hat > 0])
        threshold = np.median(z_hat[z_hat > 0]) + z_mad*z_threshold

        # outputs
        alphacsc_detection = np.zeros_like(ica_peaks)
        ica_alphacsc_aligned = np.zeros_like(ica_peaks)

        for ipk in np.where(ica_peaks > 0)[0]:
            # z_peak: max z-hat in the window
            # NOTE: window is before the ICA peak
            left_win = ipk - half_event + border
            right_win = ipk - border
            window = z_hat[left_win:right_win]
            zpk = np.argmax(window)

            if window[zpk] > threshold:
                # alignment of the ICA detections to the z-hat max peak
                # alignment to the center of the atom
                # max z-hat is at the beginning of v_hat but the spike is later
                spike_ind = left_win + zpk + half_atom_width
                alphacsc_detection[spike_ind] = 1
                ica_alphacsc_aligned[spike_ind] = ipk
        return alphacsc_detection, ica_alphacsc_aligned

    def evaluate_atoms_goodness(self, mag_data: mne.io.Raw, u_hat: np.ndarray,
                                v_hat: np.ndarray, detections: np.ndarray,
                                atoms: np.ndarray, gof: np.ndarray,
                                ) -> np.ndarray:
        """Estimate atom goodness value for each AlphaCSC atom.

        Parameters
        ----------
        mag_data : mne.io.Raw
            Full MEG recording
        u_hat : np.ndarray
            shape: atoms by channels
        v_hat : np.ndarray
            shape: atoms by times
        detections : np.ndarray
            AlphaCSC detection, binary file with the length of MEG recording
        atoms : np.ndarray
            atom that corresponds to the AlphaCSC detection
        gof : np.ndarray
            1D array with values between 0 and 1 for each atom

        Returns
        -------
        np.ndarray
            goodness value for each atom
        """
        goodness = np.zeros(self.n_atoms)
        detection_mask = detections > 0
        unique_atoms = np.unique(atoms[detection_mask])
        for atom in range(self.n_atoms):
            if atom in unique_atoms:
                atom_mask = detection_mask & (atoms == atom)
                timestamps = np.where(atom_mask)[0]
                # make epochs only with best events for this atom
                epochs = create_epochs(
                   mag_data, timestamps, tmin=-0.25, tmax=0.25)

                # Estimate goodness of the atom
                goodness[atom] = self._atom_goodness(
                   epochs, gof[atom], len(timestamps), v_hat[atom],
                   u_hat[atom])

    def _atom_goodness(self, epochs: mne.Epochs, gof: float, n_detections: int,
                       v_hat: np.ndarray, u_hat: np.ndarray) -> float:
        """Evaluate atom quality.

        Parameters
        ----------
        epochs : mne.Epochs
            [description]
        gof : float
            Goodness of dipole fitting of u_hat
        n_detections : int
            Number of detected events for this atom
        v_hat : np.ndarray
            1D array, temporal shape of the atom.
        u_hat : np.ndarray
            1D array, spatial distribution of the atom

        Returns
        -------
        float
            goodness value between 0 and 1
        """
        # First condition: absolute number of the events
        cond1 = self.atoms_selection_n_events - n_detections
        if cond1 < 0:
            cond1 = 1
        else:
            cond1 = n_detections / self.atoms_selection_n_events

        # Second condition: atoms localization GOF
        if isinstance(self.sensors, str):
            # Allow lower GOF for grad
            if self.sensors == 'grad':
                gof -= self.allow_lower_gof_grad
        cond2 = self.atoms_selection_gof - gof
        if cond2 < 0:
            cond2 = 1
        else:
            cond2 = gof / self.atoms_selection_gof

        # Third condition: v_hat and max channel pattern similarity
        # cross-correlation epochs' max channel with atom's
        # shape inside core events
        max_ch = np.argmax(u_hat)
        max_ch_waveforms = epochs.get_data()[:, max_ch, :]
        cross_corr = np.zeros(max_ch_waveforms.shape[0])
        for i, spike in enumerate(max_ch_waveforms):
            a = spike/np.linalg.norm(spike)
            a = np.abs(a)
            v = v_hat/np.linalg.norm(v_hat)
            v = np.abs(v)
            cross_corr[i] = np.correlate(a, v).mean()

        mean_cc = np.median(cross_corr)
        cond3 = self.cross_corr_threshold - mean_cc
        if cond3 < 0:
            cond3 = 1
        else:
            cond3 = mean_cc / self.cross_corr_threshold
        return (cond1 + cond2 + cond3) / 3


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
        X.to_netcdf(self.dataset, mode='a', format="NETCDF4",
                    engine="netcdf4")
        return self

    def transform(self, X) -> xr.Dataset:
        return xr.load_dataset(self.dataset)

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
