import logging
from pathlib import Path
from typing import List, Tuple, Union, Any, Dict

import mne
from mne.source_space import _check_mri

try:
    # mne 0.23
    from mne.source_space import _read_mri_info
except Exception:
    # mne 0.24
    from mne._freesurfer import _read_mri_info

import nibabel as nb
import numpy as np
from mne.fixes import _get_img_fdata
from mne.transforms import apply_trans, invert_transform
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import Delaunay
import scipy.io as sio
from sklearn.base import BaseEstimator, TransformerMixin

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

mne.set_log_level("ERROR")


def prepare_data(data: mne.io.Raw,
                 meg: Union[str, bool],
                 filtering: Union[None, List[float]],
                 resample: Union[None, float],
                 alpha_notch: Union[None, float]
                 ) -> mne.io.Raw:
    """Preprocess raw MEG data for analysis.

    Parameters
    ----------
    data : mne.io.Raw
        raw meg fif data
    meg : Union[str, bool]
        'grad', 'mag' or True
    filtering : Union[None, List[float]]
        List[highpass, lowpass, notch]
    resample : Union[None, float]
        frequency for the resampling in Hz
    alpha_notch : Union[None, float]
        alpha frequency

    Returns
    -------
    mne.io.Raw
        preprocessed raw data
    """
    data.pick_types(
        meg=meg, eeg=False, stim=False, eog=False, ecg=False,
        emg=False, misc=False)
    if filtering is not None:
        data.filter(filtering[0], filtering[1])
        data.notch_filter(filtering[2])

    if alpha_notch is not None:
        data.notch_filter(alpha_notch, trans_bandwidth=2.0)

    if resample:
        data = data.resample(resample, npad="auto")
    return data


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
    prepare_data = staticmethod(prepare_data)

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

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[Any, mne.io.Raw]) -> mne.io.Raw:
        if isinstance(X, str) or isinstance(X, Path):
            data = mne.io.read_raw_fif(X, preload=True)
        elif isinstance(X, mne.io.Raw) or isinstance(X, mne.io.RawArray):
            data = X
        else:
            data = mne.io.read_raw_fif(self.data_file, preload=True)

        data = self.prepare_data(
            data=data, meg=self.sensors, filtering=self.filtering,
            alpha_notch=self.alpha_notch, resample=self.resample)
        return data


def create_epochs(meg_data: mne.io.Raw, detections: np.ndarray,
                  tmin: float = -0.5, tmax: float = 0.5,
                  sensors: Union[str, bool] = True, ):
    '''
    Here we create epochs for events

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

    Notes
    -----
    If the difference between detections is 1 sample one of the
    events is skipped

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
        Slope components:
            t1 - baseline (20% of the slope)
            t2 - slope (50%)
            t3 - peak
    """
    # Smooth lables timeseries
    slope = gaussian_filter(label_ts, sigma=sigma)
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
    slope_left_base = max(peak - peak_width / 2, peak - 100)
    slope_times = np.linspace(max(2, slope_left_base), peak, n_points)
    return slope_times


def stc_to_nifti(stc: mne.SourceEstimate, fwd: mne.Forward,
                 subject: str, fs_sbj_dir: Union[str, Path],
                 fsave: Union[str, Path]) -> None:
    """Convert mne.SourceEstimate to NIfTI image.
    Affine transformation is in the original T1 MRI image. This is usefull for
    visual comparison results in ITK-SNAP or similar software.

    Parameters
    ----------
    stc : mne.SourceEstimate
        SourceEstimate to convert to CovexHull and save as NIfTI image
    fwd : mne.Forward
        Forward model with the same number of sources as SourceEstimate
    subject : str
        Subject name in FreeSurfer folder
    fs_sbj_dir : Union[str, Path]
        Path to the FreeSurfer folder
    fsave : Union[str, Path]
        Path to save NIfTI image
    """
    src = fwd['src']
    mri_fname = _check_mri('T1.mgz', subject, fs_sbj_dir)

    # Load the T1 data
    _, vox_mri_t, mri_ras_t, _, _, nim = _read_mri_info(
        mri_fname, units='mm', return_img=True)
    mri_vox_t = invert_transform(vox_mri_t)['trans']
    del vox_mri_t
    data = np.zeros(_get_img_fdata(nim).shape)

    lh_coordinates = src[0]['rr'][stc.lh_vertno] * 1000  # MRI coordinates
    lh_coordinates = apply_trans(mri_vox_t, lh_coordinates)
    lh_data = stc.lh_data
    lh_coordinates = lh_coordinates[lh_data[:, 0] > 0, :]
    lh_coordinates = np.int32(lh_coordinates)
    data[lh_coordinates[:, 0], lh_coordinates[:, 1], lh_coordinates[:, 2]] = 1

    rh_coordinates = src[1]['rr'][stc.rh_vertno] * 1000  # MRI coordinates
    rh_coordinates = apply_trans(mri_vox_t, rh_coordinates)
    rh_data = stc.rh_data
    rh_coordinates = rh_coordinates[rh_data[:, 0] > 0, :]
    rh_coordinates = np.int32(rh_coordinates)
    data[rh_coordinates[:, 0], rh_coordinates[:, 1], rh_coordinates[:, 2]] = 1

    test_points = np.array(np.where(data == 0)).T

    # create convex hull
    for coords in [lh_coordinates, rh_coordinates]:
        if coords.shape[0] != 0:
            dhull = Delaunay(coords)
            # find all points inside convex hull
            points_inside = test_points[
                dhull.find_simplex(test_points) >= 0]
            data[points_inside[:, 0],
                 points_inside[:, 1],
                 points_inside[:, 2]] = 1
    affine = nim.affine  # .dot(mri_ras_t['trans'])
    nb.save(nb.Nifti1Image(data, affine), fsave)


def spike_snr_all_channels(data: np.ndarray, peak):
    # data: trials, channels, times
    mean_peak = (data[:, :, peak - 20:peak + 20] ** 2).mean(axis=-1).mean(0)
    var_noise = data[:, :, :].var(axis=-1).mean(0)
    # snr = (mean_peak / var_noise).mean()
    snr = 10 * np.log10((mean_peak / var_noise).mean())
    return snr


def spike_snr_max_channel(data: np.ndarray, peak, n_max_channels=20):
    # data: trials, channels, times
    max_ch = np.argsort(
        (data[:, :, peak - 20:peak + 20] ** 2).mean(
            axis=-1).mean(axis=0))[::-1][:n_max_channels]
    mean_peak = (data[:, max_ch, peak - 20:peak + 20] ** 2).mean(axis=-1).mean(
        0)
    var_noise = data[:, max_ch, :].var(axis=-1).mean(0)
    # snr = (mean_peak / var_noise).mean()
    snr = 10 * np.log10((mean_peak / var_noise).mean())
    return snr, max_ch


def labels_to_mni(labels: List[mne.Label], fwd: mne.Forward,
                  subject: str, subjects_dir: str) -> Tuple[
    np.ndarray, np.ndarray, List[List[int]]]:
    """Convert labels to mni coordinates.

    Parameters
    ----------
    labels : List[mne.Label]
        List of a FreeSurfer/MNE label
    fwd : mne.Forward
        MNE Python forward model
    subject : str
        FreeSurfer subject name
    subjects_dir : str
        FreeSurfer subjects folder

    Returns
    -------
    np.ndarray
        array with MNI coordinates of the resection area.
    np.ndarray
        vertices like list of np.ndarray with non-zero elements for sources
        included in labels.
    List[List[int]]
        List of labels vertex indices
    """
    vertices = [i['vertno'] for i in fwd['src']]
    data = [np.zeros(len(vertices[i])) for i in [0, 1]]
    for lab in range(len(labels)):
        label = labels[lab]
        hemi = 0 if label.hemi == 'lh' else 1
        for n, i in enumerate(vertices[hemi]):
            if i in label.vertices:
                data[hemi][n] = lab + 1
    labels_mni = []
    for hemi in [0, 1]:
        l_mni = mne.vertex_to_mni(
            vertices[hemi][data[hemi] != 0],
            hemis=hemi, subject=subject,
            subjects_dir=subjects_dir)
        if l_mni.size != 0:
            labels_mni.append(l_mni)
    data = np.hstack(data)
    return np.vstack(labels_mni), data, vertices


class ToFinish(TransformerMixin, BaseEstimator):
    """Empty template to finish sklearn Pipeline."""

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        del X
        return []


def brainstorm_events_export(save_path: Path,
                             timestamps_dict: Dict[str, np.ndarray]):
    """
    Save all detections as Brainstorm events structure

    Parameters
    ----------
    save_path : pathlib.PosixPath
        path to save the file
    timestamps_dict : dict(label: timestamps in seconds)
        aligned to the first sample
    """

    def _bst_one_event(data: np.ndarray, marker_index: int, label: str,
                       color: List[float], times: List[float]):
        """
        Fill one marker in Brainstorm event's structure

        Parameters
        ----------
        data : np.array
            DESCRIPTION.
        marker_index : int
            markes' index
        label : str
            the label of the marker
        color : list, float
            len = 3
        times : list, float
            timepoints in seconds aligned to the first sample

        Returns
        -------
        arr

        """
        data[0][marker_index]['label'] = np.array([label], dtype='<U5')
        data[0][marker_index]['color'] = np.array([color])
        data[0][marker_index]['epochs'] = np.array([[1] * len(times)],
                                                   dtype=np.uint8)
        data[0][marker_index]['times'] = np.array([times])
        data[0][marker_index]['reactTimes'] = np.array([], dtype=np.uint8)
        data[0][marker_index]['select'] = np.array([[1]], dtype=np.uint8)
        data[0][marker_index]['channels'] = np.array(
            [[np.array([], dtype=np.object),
              np.array([], dtype=np.object)]],
            dtype=np.object)
        data[0][marker_index]['notes'] = np.array(
            [[np.array([], dtype=np.uint8),
              np.array([], dtype=np.uint8)]],
            dtype=np.object)
        return data

    labels = [key for key in timestamps_dict.keys()]
    dt = [('label', 'O'), ('color', 'O'), ('epochs', 'O'),
          ('times', 'O'), ('reactTimes', 'O'), ('select', 'O'),
          ('channels', 'O'), ('notes', 'O')]
    n_markers = len(labels)
    arr = np.zeros((1, n_markers), dtype=dt)
    for i in range(n_markers):
        color = np.random.rand(3, ).tolist()
        times = timestamps_dict[labels[i]]
        times = np.unique(times)
        arr = _bst_one_event(
            arr, i, labels[i], color, times.tolist())
    sio.savemat(save_path, {'events': arr})


def read_dip(dip_path: Path, sfreq: int = 1000) -> np.ndarray:
    """
    Fix header and read dipoles
    Returns
    -------
    dipoles : np.ndarray
        timestamps in milliseconds

    See Also https://mne.tools/stable/generated/mne.read_dipole.html
    """

    def _replace_txt_in_dip(file_path, pattern, subst):
        """
        Write the content to a new file and replaces the old file with the new
        file.
        """

        # Create temp file
        fh, abs_path = mkstemp()
        with fdopen(fh, 'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    if line.find(subst) > 0:
                        new_file.write(line)
                    else:
                        new_file.write(line.replace(pattern, subst))
        # Copy the file permissions from the old file to the new file
        copymode(file_path, abs_path)
        # Remove original file
        remove(file_path)
        # Move new file
        move(abs_path, file_path)

    # Add '/ms' to begin header
    _replace_txt_in_dip(dip_path, 'begin', 'begin/ms')

    # Read dipoles
    logging.info(f'Reading manual detections from {str(dip_path)}')
    dipoles = sorted(mne.read_dipole(str(dip_path), verbose='debug').times)
    dipoles = np.array(dipoles) * sfreq
    dipoles = np.sort(np.unique(np.rint(dipoles)))
    logging.debug(f'Dipole times in milliseconds: {dipoles}')
    return dipoles
