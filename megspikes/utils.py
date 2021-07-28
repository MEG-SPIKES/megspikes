# -*- coding: utf-8 -*-
from typing import Union, List, Tuple, Any
from pathlib import Path
import mne
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


class PrepareData(BaseEstimator, TransformerMixin):
    """Prepare mne.io.Raw object analysis

    Parameters
    ----------
    data_file : str
        path to meg fif data
    sensors : str or bool
        'grad', 'mag' or True
    resample : int, optional
        downsample data to the resample value, by default None
    filtering : list of integers, optional
        highpass, lowpass, notch
    data : mne.io.Raw, optional
        filter and/or resample not default (fif_file)
        data, by default None
    alpha_notch : bool, optional
        apply 8-12Hz notch filter, by default False


    Returns
    -------
    mne.io.Raw

    """
    def __init__(self,
                 data_file: Union[str, Path, None] = None,
                 sensors: Union[str, bool] = True,
                 filtering: Union[None, List[float]] = [2, 90, 50],
                 resample: Union[None, float] = None,
                 alpha_notch: Union[None, float] = None) -> None:
        self.data_file = data_file
        self.sensors = sensors
        self.filtering = filtering
        self.resample = resample
        self.alpha_notch = alpha_notch

    def fit(self, X: Union[Any, Tuple[xr.Dataset, mne.io.Raw]], y=None):
        return self

    def transform(self, X: Union[Any, Tuple[xr.Dataset, mne.io.Raw]],
                  ) -> Tuple[Any, mne.io.Raw]:
        if isinstance(X, tuple):
            data = self._prepare_data(data=X[1])
            return (X[0], data)
        else:
            data = self._prepare_data(data=None)
            return (X, data)

    def _prepare_data(self, data: Union[None, mne.io.Raw]) -> mne.io.Raw:
        if data is None:
            data = mne.io.read_raw_fif(self.data_file, preload=True)
        data.pick_types(
            meg=self.sensors, eeg=False, stim=False, eog=False, ecg=False,
            emg=False, misc=False)
        if self.filtering is not None:
            data.filter(self.filtering[0], self.filtering[1])
            data.notch_filter(self.filtering[2])

        if self.alpha_notch:
            data.notch_filter(self.alpha_notch, trans_bandwidth=2.0)

        if self.resample:
            data = data.resample(self.resample, npad="auto")
        return data


def create_epochs(meg_data: mne.io.Raw, detections: np.ndarray,
                  tmin: float = -0.5, tmax: float = 0.5,
                  sensors: Union[str, bool] = True,):
    '''
    Here we create epochs for events
    NOTE: !!! if the difference between detections is 1 sample one of the
    events is skipped

    Parameters
    ----------
    meg_data : fif
        entire record from which events will be retrieved
    detections : list
        timepoints in ms. Should be aligned to the first sample.
        timepoints should be sorted
    tmin : float, optional
        time before the detection. The default is -0.5.
    tmax : float, optional
        time after the detection. The default is 0.5.
    sensors : str or bool, optional
        channels type ("gard", "mag" or True). The default is True.

    Returns
    -------
    epochs : MNE epochs
        Preloaded epochs for each detected event.

    '''
    meg_data.load_data()
    new_events, eve = [], []

    for spike_time in detections:
        eve = [int(round(spike_time)), 0, 1]
        new_events.append(eve)

    # Adding new stimulus channel
    ch_name = 'NEW_DET'
    if ch_name not in meg_data.info['ch_names']:
        stim_data = np.zeros((1, len(meg_data.times)))
        info_sp = mne.create_info(
            [ch_name], meg_data.info['sfreq'], ['stim'])
        stim_sp = mne.io.RawArray(stim_data, info_sp)
        meg_data.add_channels([stim_sp], force_update_info=True)

    # Adding events
    meg_data.add_events(new_events, stim_channel=ch_name, replace=True)
    events = mne.find_events(meg_data, stim_channel=ch_name)
    event_id = {'DET': 1}
    picks = mne.pick_types(
        meg_data.info, meg=sensors, eeg=False, eog=False)
    epochs = mne.Epochs(meg_data, events, event_id, tmin, tmax, baseline=None,
                        picks=picks, preload=True)
    del meg_data, picks, event_id
    return epochs


def onset_slope_timepoints(label_ts: np.ndarray,
                           n_points: int = 3,
                           sigma: float = 3,
                           peaks_width: float = 20,
                           peaks_rel_hight: float = 0.6
                           ) -> np.ndarray:
    """ Find the peak of the spike, 50% and 20% of the slope.
    """
    # Smooth lables timeseries
    slope = gaussian_filter(label_ts[0].mean(axis=0), sigma=sigma)
    # Find all peaks TODO:, wlen=100
    peaks, properties = signal.find_peaks(slope, width=peaks_width)
    assert len(peaks) > 0, "No peaks detected"
    # Find widths of the peaks using relative hight
    widths_full = signal.peak_widths(slope, peaks, rel_height=peaks_rel_hight)
    # Sort peaks by prominences
    peak_ind = np.argmax(properties['prominences'].flatten())
    # slope_beginig = properties['left_bases'][peak_ind]
    peak_width = widths_full[0][peak_ind]
    peak = peaks[peak_ind]
    # if left base is too far use 100 samples
    slope_left_base = max(peak - peak_width / 2, peak-100)
    slope_times = np.linspace(max(2, slope_left_base), peak, n_points)
    return slope_times


class ToFinish(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        del X
        return []
