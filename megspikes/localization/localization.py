import logging
from typing import Any, List, Tuple, Union
import warnings

import mne
import numpy as np
import xarray as xr
from mne.beamformer._compute_beamformer import _prepare_beamformer_input
from mne.io.pick import pick_channels_forward, pick_info
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.morph import _get_subject_sphere_tris, _hemi_morph
from mne.surface import mesh_edges
from mne.utils import _check_info_inv
from scipy import linalg, sparse
from sklearn.base import BaseEstimator, TransformerMixin

from ..casemanager.casemanager import CaseManager
from ..database.database import (check_and_read_from_dataset,
                                 check_and_write_to_dataset)
from ..utils import create_epochs, onset_slope_timepoints

mne.set_log_level("ERROR")


def array_to_stc(data: np.ndarray, fwd: mne.Forward, subject: str
                 ) -> mne.SourceEstimate:
    """Convert SourceEstimate data to mne.SourceEstimate object.

    Parameters
    ----------
    data : np.ndarray
        1D array with the length equal the number of sources in the head model
    fwd : mne.Forward
        Head model
    subject : str
        Subject (case) name

    Returns
    -------
    mne.SourceEstimate
    """
    vertices = [i['vertno'] for i in fwd['src']]
    return mne.SourceEstimate(
        data, vertices, tmin=0, tstep=0.001, subject=subject)


class Localization():
    """Base class to prepare components for source localization and evaluation.
    """
    array_to_stc = staticmethod(array_to_stc)

    def setup_fwd(self, case: CaseManager, sensors: Union[str, bool] = True,
                  spacing: str = 'oct5') -> None:
        """Prepare objects for source localization.

        Parameters
        ----------
        case : CaseManager
            custom class that includes forward model, FreeSurfer folder,
            raw.Info from the preprocessing step
        sensors : Union[str, bool], optional
            magnetometers, gradiometers or both, by default True
        spacing : str, optional
            The number of sources in the forward model. Currently accepted two
            options: oct5 - low resolution; ico5 - high resolution. Forward
            model is precomputed in the earlier steps in the CaseManager,
            by default 'oct5'

        Raises
        ------
        RuntimeError
            Forward model should be included in the CaseManager.
        """
        if not isinstance(case.fwd[spacing], mne.Forward):
            raise RuntimeError("CaseManager don't include forward model")
        self.sensors = sensors
        self.case = case
        self.case_name = case.case
        self.bem = case.bem[spacing]
        self.trans = case.trans[spacing]
        self.freesurfer_dir = case.freesurfer_dir
        self.info = case.info
        self.info, self.fwd, self.cov = self.pick_sensors(
            case.info, case.fwd[spacing], sensors)

    def pick_sensors(self, info: mne.Info, fwd: mne.Forward,
                     sensors: Union[str, bool] = True
                     ) -> Tuple[mne.Info, mne.Forward, mne.Covariance]:
        """Pick one sensors type from mne.Info and mne.Forward.

        Parameters
        ----------
        info : mne.Info
            info of the MEG data file
        fwd : mne.Forward
            valid forward model
        sensors : Union[str, bool], optional
            selected type of the sensors: 'grad', 'mag' or True,
            by default True

        Returns
        -------
        Tuple[mne.Info, mne.Forward, mne.Covariance]
            Info, Forward model and Diagonal covariance for the for the
            selected sensors type
        """
        info_ = mne.pick_info(info, mne.pick_types(info, meg=sensors))
        if isinstance(sensors, str):
            fwd_ = mne.pick_types_forward(fwd, meg=sensors)
        else:
            fwd_ = fwd
        cov = mne.make_ad_hoc_cov(info_)
        return info_, fwd_, cov

    def make_labels_ts(self, stc: mne.SourceEstimate,
                       inverse_operator: mne.minimum_norm.InverseOperator,
                       mode: str = 'mean') -> np.ndarray:
        """Extract anatomical labels time courses.

        Parameters
        ----------
        stc : mne.SourceEstimate
            [description]

        Returns
        -------
        label_tc : np.ndarray
            with the shape: lables by time
        """
        labels_parc = mne.read_labels_from_annot(
            subject=self.case_name,  subjects_dir=self.freesurfer_dir)
        src = inverse_operator['src']

        label_ts = mne.extract_label_time_course(
            [stc], labels_parc, src, mode=mode, allow_empty=True)
        return label_ts

    def binarize_stc(self, data: np.ndarray, fwd: mne.Forward,
                     smoothing_steps: int = 3,
                     amplitude_threshold: float = 0.5,
                     min_sources: int = 10,
                     normalize: bool = True) -> np.ndarray:
        """Binarization and smoothing of one SourceEstimate timepoint.
           converting to mne.SourceEstimate.

        Parameters
        ----------
        data : np.ndarray
            1d, length is equal the number of fwd sources
        fwd : mne.Forward
            subject's head model
        smoothing_steps : int, optional
            smoothing individual spikes and final results, by default 3
        amplitude_threshold : float, optional
            select sources with the amplitude above amplitude_threshold of the
            maximum amplitude, by default 0.5
        min_sources : int, optional
            select at least min_sources even if their amplitude below
            amplitude_threshold, by default 10
        normalize : bool, optional
            normalize clusters source estimate using peak amplitude; if False
            then slope and baseline predictions are normalized according slope
            and baseline maximum values respectively, by default True

        Returns
        -------
        np.ndarray
            binarized array with the same shape as input data

        Notes
        -----
        We follow procedure described in [1]_.

        References
        ----------
        .. [1] Tanaka, N., Papadelis, C., Tamilia, E., Madsen, J. R., Pearl, P.
            L., & Stufflebeam, S. M. (2018). Magnetoencephalographic Mapping of
            Epileptic Spike Population Using Distributed Source Analysis:
            Comparison With Intracranial Electroencephalographic Spikes.
            Journal of Clinical Neurophysiology, 35(4), 339–345.
            https://doi.org/10.1097/WNP.0000000000000476

        """
        vertices = [i['vertno'] for i in fwd['src']]

        # normalize data
        if normalize:
            data /= data.max()

        # check if amplitude is too small
        if np.sum(data > amplitude_threshold) < min_sources:
            sort_data_ind = np.argsort(data)[::-1]
            data[sort_data_ind[:min_sources]] = 1

        # threshold the data
        data[data < amplitude_threshold] = 0
        data[data >= amplitude_threshold] = 1

        # Create SourceEstimate object
        stc = mne.SourceEstimate(data, vertices, tmin=0,
                                 tstep=0.001, subject=self.case_name)
        # NOTE: final_data is 1D here
        final_data = np.zeros_like(data)

        # Smooth final surface left hemi
        lh_data = self._smooth_binarized_stc(
            stc, hemi_idx=0, smoothing_steps=smoothing_steps)
        final_data[:len(stc.vertices[0])] = lh_data

        # Smooth final surface right hemi
        rh_data = self._smooth_binarized_stc(
            stc, hemi_idx=1, smoothing_steps=smoothing_steps)
        final_data[len(stc.vertices[0]):] = rh_data
        final_data[final_data > 0] = 1

        # return binary map
        return final_data

    def _smooth_binarized_stc(self, stc: mne.SourceEstimate, hemi_idx: int,
                              smoothing_steps: int = 10):
        """Smooth binary SourceEstimate. """
        vertices = stc.vertices[hemi_idx]
        tris = _get_subject_sphere_tris(
            self.case_name, self.freesurfer_dir)[hemi_idx]
        e = mesh_edges(tris)
        n_vertices = e.shape[0]
        maps = sparse.identity(n_vertices).tocsr()

        if hemi_idx == 0:
            data = stc.data[:len(stc.vertices[hemi_idx]), :]
        else:
            data = stc.data[len(stc.vertices[0]):, :]
        smooth_mat = _hemi_morph(
            tris, vertices, vertices, smoothing_steps, maps, warn=False)
        data = smooth_mat.dot(data)
        return data.flatten()


class ICAComponentsLocalization(Localization, BaseEstimator, TransformerMixin):
    """ Localize ICA components using mne.fit_dipole()."""
    def __init__(self, case: CaseManager, sensors: Union[str, bool] = True,
                 spacing: str = 'oct5'):
        self.spacing = spacing
        self.setup_fwd(case, sensors, spacing)

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        # read from database ica components
        components = check_and_read_from_dataset(X[0], 'ica_components')
        components = components.T
        # create EvokedArray
        evoked = mne.EvokedArray(components, self.info)

        dip = mne.fit_dipole(evoked, self.cov, self.bem, self.trans)[0]

        locs = np.zeros((len(dip), 3))
        gof = np.zeros(len(dip))
        for n, d in enumerate(dip):
            locs[n, :] = mne.head_to_mni(
                d.pos[0],  self.case_name, self.fwd['mri_head_t'],
                subjects_dir=self.freesurfer_dir)
            gof[n] = d.gof[0]

        check_and_write_to_dataset(
            X[0], 'ica_component_properties', locs,
            dict(ica_component_property=['mni_x', 'mni_y', 'mni_z']))
        check_and_write_to_dataset(
            X[0], 'ica_component_properties', gof,
            dict(ica_component_property='gof'))
        logging.info("ICA components are localized.")
        return X


class PeakLocalization(Localization, BaseEstimator, TransformerMixin):
    """Source reconsturction using RAP MUSIC algorithm

    Parameters
    ----------
    sfreq : int, optional
        downsample freq, by default 200.
    window : list, optional
        MUSIC window, by default [-20, 30]
    """
    def __init__(self, case: CaseManager, sensors: Union[str, bool] = True,
                 sfreq: int = 200, window: List[int] = [-20, 30],
                 spacing: str = 'oct5'):
        self.spacing = spacing
        self.setup_fwd(case, sensors, spacing)
        self.window = window
        self.sfreq = sfreq

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        detection = check_and_read_from_dataset(
            X[0], 'detection_properties',
            dict(detection_property='ica_detection'))
        # samples
        timestamps = np.where(detection > 0)[0]

        data = X[1].get_data()
        window = (np.array(self.window)/1000)*self.sfreq
        mni_coords, subcorr = self.fast_music(
            data, self.info, timestamps, window=window)

        full_subcorrs = np.zeros_like(detection)
        full_subcorrs[detection > 0] = subcorr
        check_and_write_to_dataset(
            X[0], 'detection_properties', full_subcorrs,
            dict(detection_property='subcorr'))

        full_coords = np.zeros((detection.shape[0], 3))
        full_coords[detection > 0, :] = mni_coords
        check_and_write_to_dataset(
            X[0], 'detection_properties', full_coords[:, 0],
            dict(detection_property='mni_x'))
        check_and_write_to_dataset(
            X[0], 'detection_properties', full_coords[:, 1],
            dict(detection_property='mni_y'))
        check_and_write_to_dataset(
            X[0], 'detection_properties', full_coords[:, 2],
            dict(detection_property='mni_z'))
        logging.info("ICA peaks are localized.")
        return X

    def fast_music(self, data: np.ndarray, info: mne.Info, spikes: np.ndarray,
                   window: List[int]):
        """
        :Authors:
        ------
            Daria Kleeva <dkleeva@gmail.com>
        """
        common_atr = self._prepare_rap_music_input(info, self.cov, self.fwd)
        zyx_mni = np.zeros((len(spikes), 3))
        subcorrs = np.zeros(len(spikes), dtype=np.float64)

        for n, spike in enumerate(spikes):
            spike_data = data[:, int(spike+window[0]): int(spike+window[1])]
            pos, ori, subcorr = self._apply_music(
                *common_atr, data=spike_data, n_dipoles=5)
            zyx_mni[n] = mne.head_to_mni(
                pos,  self.case_name, self.fwd['mri_head_t'],
                subjects_dir=self.freesurfer_dir)
            subcorrs[n] = subcorr
        return zyx_mni, subcorrs

    def _prepare_rap_music_input(self, info: mne.Info,
                                 noise_cov: mne.Covariance,
                                 forward: mne.Forward):
        """
        :Authors:
        ------
            Daria Kleeva <dkleeva@gmail.com>
        """
        picks = _check_info_inv(
            info, forward, data_cov=None, noise_cov=noise_cov)
        info = pick_info(info, picks)
        is_free_ori, info, _, _, G, whitener, _, _ = _prepare_beamformer_input(
            info, forward, noise_cov=noise_cov, rank=None)
        forward = pick_channels_forward(
            forward, info['ch_names'], ordered=True)
        del info

        n_orient = 3 if is_free_ori else 1

        Ug, Sg, Vg = [], [], []

        # G3->G2
        if n_orient == 3:  # or other optional condition
            G3 = G.copy()
            G = np.zeros((G3.shape[0], (G3.shape[1]//n_orient)*2))
            for i_source in range(G3.shape[1]//n_orient):
                idx_k = slice(n_orient * i_source, n_orient * (i_source + 1))
                idx_r = slice(2 * i_source, 2 * (i_source + 1))
                Gk = G3[:, idx_k]
                Gk = np.dot(Gk, forward['source_nn'][idx_k])
                U, S, V = linalg.svd(Gk, full_matrices=True)
                G[:, idx_r] = U[:, 0:2]
            n_orient = 2

        # to compute subcorrs
        for i_source in range(G.shape[1] // n_orient):
            idx_k = slice(n_orient * i_source, n_orient * (i_source + 1))
            Gk = G[:, idx_k]
            if (n_orient == 3):
                Gk = np.dot(Gk, forward['source_nn'][idx_k])
            _Ug, _Sg, _Vg = linalg.svd(Gk, full_matrices=False)
            # Now we look at the actual rank of the forward fields
            # in G and handle the fact that it might be rank defficient
            # eg. when using MEG and a sphere model for which the
            # radial component will be truly 0.
            rank = np.sum(_Sg > (_Sg[0] * 1e-6))
            if rank == 0:
                # return 0, np.zeros(len(G))
                Ug.append(0)
                Sg.append(np.zeros(len(G)))
                Vg.append(0)
            else:
                rank = max(rank, 2)  # rank cannot be 1
                Ug.append(_Ug[:, :rank].T.conjugate())
                Sg.append(_Sg[:rank].T.conjugate())
                Vg.append(_Vg[:rank].T.conjugate())
        Ug = np.asarray(Ug)
        Ug = np.reshape(Ug, (Ug.shape[0]*Ug.shape[1], Ug.shape[2]))

        return picks, forward, whitener, is_free_ori, G, Ug, Sg, Vg, n_orient

    def _apply_music(self, picks, forward, whitener, is_free_ori,
                     G, Ug, Sg, Vg, n_orient, data, n_dipoles=5):
        """RAP-MUSIC for evoked data.
        Parameters
        ----------
        forward : instance of Forward
            Forward operator.
        n_dipoles : int
            The number of dipoles to estimate. The default value is 2.
        Returns
        -------
        dipoles : list of instances of Dipole
            The dipole fits.
        explained_data : array | None
            Data explained by the dipoles using a least square fitting with the
            selected active dipoles and their estimated orientation.
            Computed only if return_explained_data is True.
        :Authors:
        ------
            Daria Kleeva <dkleeva@gmail.com>
        """
        # whiten the data (leadfield already whitened)
        data = data[picks]
        data = np.dot(whitener, data)

        eig_values, eig_vectors = linalg.eigh(np.dot(data, data.T))
        phi_sig = eig_vectors[:, -n_dipoles:]

        phi_sig_proj = phi_sig.copy()

        tmp = np.dot(Ug, phi_sig_proj)

        tmp2 = np.multiply(tmp, tmp.conj()).transpose()
        # find off-diagonals
        tmp2d = np.multiply(tmp[::2, :], tmp[1::2, :].conj())
        tmp2_11_22 = np.sum(tmp2, axis=0)  # find diagonals
        tmp2_11_22 = np.reshape(
            tmp2_11_22, (tmp2_11_22.shape[0]//2, 2)).transpose()
        tmp2_12_21 = np.sum(tmp2d, axis=1)  # find off-diagonals
        T = (np.sum(tmp2_11_22, axis=0))  # trace
        D = np.prod(tmp2_11_22, axis=0)-np.multiply(
            tmp2_12_21, tmp2_12_21.conj())  # determinant
        # apply theorem about eigenvalues
        Covar = 0.5*(T+np.sqrt(np.multiply(T, T)-4*D))
        Covar = np.sqrt(Covar)
        subcorr_max = Covar.max()

        source_ori = forward['source_nn'][np.argmax(Covar)]
        source_pos = forward['source_rr'][np.argmax(Covar)]
        return source_pos, source_ori, subcorr_max


class AlphaCSCComponentsLocalization(Localization, BaseEstimator,
                                     TransformerMixin):
    def __init__(self, case: CaseManager, sensors: Union[str, bool] = True,
                 spacing: str = 'oct5'):
        self.spacing = spacing
        self.setup_fwd(case, sensors, spacing)

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        components = check_and_read_from_dataset(X[0], 'alphacsc_u_hat')
        components = components.T
        evoked = mne.EvokedArray(components, self.info)
        dip = mne.fit_dipole(evoked, self.cov, self.bem, self.trans)[0]

        locs = np.zeros((len(dip), 3))
        gof = np.zeros(len(dip))
        for n, d in enumerate(dip):
            locs[n, :] = mne.head_to_mni(
                d.pos[0],  self.case_name, self.fwd['mri_head_t'],
                subjects_dir=self.freesurfer_dir)
            gof[n] = d.gof[0]

        check_and_write_to_dataset(
            X[0], 'alphacsc_atoms_properties', locs,
            dict(alphacsc_atom_property=['mni_x', 'mni_y', 'mni_z']))
        check_and_write_to_dataset(
            X[0], 'alphacsc_atoms_properties', gof,
            dict(alphacsc_atom_property='gof'))
        logging.info("AlphaCSC components are localized.")
        return X


class ClustersLocalization(Localization, BaseEstimator, TransformerMixin):
    def __init__(self, case: CaseManager,
                 inv_method: str = 'MNE',
                 epochs_window: Tuple[float] = (-0.5, 0.5),
                 spacing='ico5'):
        self.setup_fwd(case, sensors=True, spacing=spacing)
        self.inv_method = inv_method
        self.epochs_window = epochs_window
        self.spacing = spacing

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        assert X[0].time.attrs['sfreq'] == X[1].info['sfreq'], (
            "Wrong sfreq of the fif file or database time coordinate")
        spikes = check_and_read_from_dataset(
            X[0], 'spike', dict(detection_property='detection'))
        detection_mask = spikes > 0
        clusters = check_and_read_from_dataset(
            X[0], 'spike', dict(detection_property='cluster'))
        sensors = check_and_read_from_dataset(
            X[0], 'cluster_properties', dict(cluster_property='sensors'),
            dtype=np.int64)

        all_clusters = np.int32(np.unique(clusters[detection_mask]))
        clusters_properties = np.zeros((len(all_clusters), 3))
        n_times = len(X[0].time_evoked.values)

        # Select only the relevant channels
        X = (self.select_only_meg_channels(X[0]), X[1])

        evokeds = []
        for cluster in all_clusters:
            evoked = self.average_cluster(
                X[1], detection_mask, clusters, cluster)
            check_and_write_to_dataset(
                X[0], 'evoked', evoked.data[:, :n_times],
                dict(cluster=cluster))
            evokeds.append(evoked)

        inverse_operator = {}
        for sensor_ind in np.unique(sensors):
            sensor_type = 'grad' if sensor_ind == 0 else 'mag'
            assert sensor_type in X[0].sensors.values, (
                f"Sensors with the type {sensor_type} are not in the sensors"
                f"coordinates ({X[0].sensors.values}) in the database.")
            info, fwd, cov = self.pick_sensors(
                self.info, self.fwd, sensor_type)
            inverse_operator[sensor_type] = make_inverse_operator(
                info, fwd, cov, depth=None, fixed=False)

            for n, cluster in enumerate(all_clusters):
                evoked_sens = evokeds[n].copy().pick_types(meg=sensor_type)
                # minimum norm
                stc, label_ts = self.minimum_norm(
                    evoked_sens, inverse_operator[sensor_type])
                check_and_write_to_dataset(
                    X[0], 'mne_localization', stc.data[:, :n_times],
                    dict(sensors=sensor_type, cluster=cluster))
                if sensors[n] == sensor_ind:
                    clusters_properties[n, :] = onset_slope_timepoints(
                        label_ts[0].mean(axis=0))

        check_and_write_to_dataset(
            X[0], 'cluster_properties', clusters_properties, dict(
                cluster_property=['time_baseline', 'time_slope', 'time_peak']))
        logging.info("Clusters are localized.")
        return X

    def minimum_norm(self, evoked: Union[mne.Evoked, mne.EvokedArray],
                     inverse_operator: mne.minimum_norm.InverseOperator,
                     inv_method: str = 'MNE') -> Tuple[
                         mne.SourceEstimate, np.ndarray]:
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        stc = apply_inverse(
            evoked, inverse_operator, lambda2, inv_method, pick_ori=None)
        label_ts = self.make_labels_ts(stc, inverse_operator)
        return stc, label_ts

    def average_cluster(self, meg_data: mne.io.Raw, detection_mask: np.ndarray,
                        clusters: np.ndarray, cluster: int) -> mne.Evoked:
        # select timestamps for the cluster
        cluster_mask = detection_mask & (clusters == cluster)
        times = np.where(cluster_mask)[0]
        # add first sample
        times += meg_data.first_samp
        # Create epochs for the cluster
        epochs = create_epochs(
            meg_data, times, tmin=self.epochs_window[0],
            tmax=self.epochs_window[1])
        # Create Evoked
        return epochs.average()

    def select_only_meg_channels(self, ds: xr.Dataset):
        """Select only the relevant channels in the dataset
           FIXME Temporary solution to avoid wrong dimensions error
        """
        sensors = ds.sensors.values.tolist()
        channels = []
        for sens in sensors:
            channels += ds.channel_names.attrs[sens].tolist()
        return ds.loc[dict(sensors=sensors, channel=channels)]


class ForwardToMNI(Localization, BaseEstimator, TransformerMixin):
    """Save MNI coordinates of all Forward model sources.
    """
    def __init__(self, case: CaseManager, spacing='ico5'):
        self.setup_fwd(case, sensors=True, spacing=spacing)
        self.spacing = spacing

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        fwd_source_coords = []
        for hemi in [0, 1]:
            hemi_mni = mne.vertex_to_mni(
                self.fwd['src'][hemi]['vertno'], hemis=hemi,
                subject=self.case_name, subjects_dir=self.freesurfer_dir)
            fwd_source_coords.append(hemi_mni)
        fwd_source_coords = np.vstack(fwd_source_coords)
        check_and_write_to_dataset(
            X[0], 'fwd_mni_coordinates', fwd_source_coords)
        return X


class PredictIZClusters(Localization, BaseEstimator, TransformerMixin):
    """Predict irritative area.

    See Also
    --------
    mne.SourceEstimate

    Parameters
    ----------
    case : CaseManager
        [description]
    sensors : Union[str, bool], optional
        [description], by default True
    smoothing_steps_one_cluster : int, optional
        amount of smoothing for the individual clusters binary map,
        by default 3
    smoothing_steps_final : int, optional
        amount of smoothing for the final prediction, by default 10
    amplitude_threshold : float, optional
        amplitude threshold for the SourceEstimate binarization; Note that the
        amplitude values are between 0 and 1 because data are normalized., by
        default 0.5
    min_sources : int, optional
        select at least min_sources sources it there is less sources above the
        amplitude_threshold, by default 10
    normalize_using_peak : bool, optional
        normalize clusters source estimate using peak amplitude; if False then
        slope and baseline predictions are normalized according slope and
        baseline maximum values respectively, by default True
    spacing : str, optional
        the number of sources in the forward model, by default 'ico5'
    """
    def __init__(self,
                 case: CaseManager,
                 sensors: Union[str, bool] = True,
                 smoothing_steps_one_cluster: int = 3,
                 smoothing_steps_final: int = 10,
                 amplitude_threshold: float = 0.5,
                 min_sources: int = 10,
                 normalize_using_peak: bool = True,
                 spacing='ico5'):
        self.setup_fwd(case, sensors, spacing=spacing)
        self.spacing = spacing
        self.smoothing_steps_one_cluster = smoothing_steps_one_cluster
        self.smoothing_steps_final = smoothing_steps_final
        self.amplitude_threshold = amplitude_threshold
        self.min_sources = min_sources
        self.normalize_using_peak = normalize_using_peak

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        clusters = check_and_read_from_dataset(
            X[0], 'cluster_properties', dict(cluster_property=['cluster_id']),
            dtype=np.int64)
        sensors = check_and_read_from_dataset(
            X[0], 'cluster_properties', dict(cluster_property=['sensors']),
            dtype=np.int64)
        baseline = check_and_read_from_dataset(
            X[0], 'cluster_properties',
            dict(cluster_property=['time_baseline']), dtype=np.int64)
        slope = check_and_read_from_dataset(
            X[0], 'cluster_properties', dict(cluster_property=['time_slope']),
            dtype=np.int64)
        peak = check_and_read_from_dataset(
            X[0], 'cluster_properties', dict(cluster_property=['time_peak']),
            dtype=np.int64)
        selected_for_iz_prediction = check_and_read_from_dataset(
            X[0], 'cluster_properties',
            dict(cluster_property=['selected_for_iz_prediction'])),
        stc_clusters = check_and_read_from_dataset(
            X[0], 'mne_localization')

        selected_clusters = selected_for_iz_prediction[0].flatten()
        if (selected_clusters == 0).all():
            warnings.warn('No clusters selected for IZ prediction.'
                          'All clusters were selected instead.')
            selected_clusters += 1
        selected_clusters = selected_clusters != 0

        # Normalize source estimate using peak amplitude
        if self.normalize_using_peak:
            for i in range(stc_clusters.shape[0]):
                for ii in range(stc_clusters.shape[1]):
                    stc_clusters[i, ii, :, :] /= stc_clusters[
                        i, ii, :, peak[ii]].max()

        for slope_time, slope_name in zip([baseline, slope, peak],
                                          ['baseline', 'slope', 'peak']):
            iz_prediciton = self.make_iz_prediction(
                stc_clusters,  slope_time, clusters, sensors,
                selected_clusters)
            check_and_write_to_dataset(
                X[0], 'iz_prediction', iz_prediciton, dict(
                    iz_prediction_timepoint=slope_name))
        logging.info("Irritative zone prediction using clusters is finished.")
        return X

    def make_iz_prediction(self, stc_clusters: np.ndarray,
                           slope_time: np.ndarray,
                           clusters: np.ndarray,
                           sensors: np.ndarray,
                           selected_clusters: np.ndarray
                           ) -> np.ndarray:
        """Predict irritative zone using clusters MNE localization.

        Parameters
        ----------
        stc_clusters : np.ndarray
            MNE clusters' source estimate data with the shape:
            number of sensors, number clusters, number of sources, times
        slope_time : np.ndarray
            the list of timepoints at which prediction should be made;
            length is the same as the number of clusters
        clusters : np.ndarray
            array with cluster indices
        sensors : np.ndarray
            list of sensors indices for each cluster
        selected_clusters : np.ndarray
            bool array whether the cluster was selected or not

        Returns
        -------
        np.ndarray
            1D binary array with the length equal the number of sources.
            1 means that the source was selected as an irritative area.
        """
        normalize_stc = False if self.normalize_using_peak else True
        clusters_stcs = []
        n_clusters = sum(selected_clusters)
        for i, (cluster, sens) in enumerate(zip(clusters, sensors)):
            if selected_clusters[i]:  # skip if the class is not selected
                stc_cluster = stc_clusters[
                    sens, cluster, :, slope_time[i]].squeeze()
                # Binarize SourceEstimate
                stc_cluster_bin = self.binarize_stc(
                    stc_cluster, self.fwd, self.smoothing_steps_one_cluster,
                    self.amplitude_threshold, self.min_sources,
                    normalize_stc)
                clusters_stcs.append(stc_cluster_bin)

        # Binarize stc again
        iz_prediciton = np.stack(clusters_stcs, axis=-1).sum(axis=-1)
        iz_prediciton[iz_prediciton < n_clusters/2] = 0
        iz_prediciton[iz_prediciton >= n_clusters/2] = 1
        return self.binarize_stc(
            iz_prediciton, self.fwd, self.smoothing_steps_final,
            self.amplitude_threshold, self.min_sources)


class ManualEventsLocalization(Localization, BaseEstimator, TransformerMixin):
    """Predict irritative zone using localizations of manually detected events.

    Parameters
    ----------
    case : CaseManager
        [description]
    inv_method : str, optional
        by default 'MNE'
    epochs_window : Tuple[float], optional
        [description]
    spacing : str, optional
        the number of sources in the forward model, by default 'ico5'
    sensors : Union[str, bool], optional
        [description], by default 'grad'
    smoothing_steps : int, optional
        amount of smoothing for the individual spike binary map,
        by default 10
    smoothing_steps_final : int, optional
        amount of smoothing for the final prediction, by default 10

    References
    ----------
    .. [1] Tanaka, N., Papadelis, C., Tamilia, E., Madsen, J. R., Pearl, P.
        L., & Stufflebeam, S. M. (2018). Magnetoencephalographic Mapping of
        Epileptic Spike Population Using Distributed Source Analysis:
        Comparison With Intracranial Electroencephalographic Spikes.
        Journal of Clinical Neurophysiology, 35(4), 339–345.
        https://doi.org/10.1097/WNP.0000000000000476
    """

    def __init__(self, case: CaseManager,
                 inv_method: str = 'MNE',
                 epochs_window: Tuple[float] = (-0.5, 0.5),
                 spacing='ico5',
                 sensors='grad',
                 smoothing_steps=10,
                 smoothing_steps_final=10):

        self.setup_fwd(case, sensors=sensors, spacing=spacing)
        self.inverse_operator = make_inverse_operator(
            self.info, self.fwd, self.cov, depth=None, fixed=False)
        self.inv_method = inv_method
        self.epochs_window = epochs_window
        self.spacing = spacing
        self.smoothing_steps = smoothing_steps
        self.smoothing_steps_final = smoothing_steps_final

    def fit(self, X: Tuple[np.ndarray, mne.io.Raw], y=None):
        return self

    def transform(self, X) -> np.ndarray:
        # add first sample
        times = X[0] + X[1].first_samp

        epochs = create_epochs(X[1], times, tmin=-0.25, tmax=0.25)
        stc_manual = self.epochs_to_stc(epochs)
        return stc_manual

    def epochs_to_stc(self, epochs):
        """Convert epochs to source estimate
        Parameters
        ----------
        epochs : mne.Epochs
            manual spikes epochs
        Returns
        -------
        stc : mne.SourceEstimate
            final source estimate for all epochs
        """
        n_epochs, n_channels, n_times = epochs.get_data().shape
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs, self.inverse_operator, lambda2,
            self.inv_method, pick_ori=None)

        results = []
        for stc in stcs:
            stc_spike_bin = self.binarize_stc(
                stc.data[:, n_times//2].squeeze(),
                self.fwd, smoothing_steps=self.smoothing_steps)
            results.append(stc_spike_bin)

        # Binarize stc again
        iz_prediction = np.stack(results, axis=-1).sum(axis=-1)
        iz_prediction[iz_prediction < n_epochs/2] = 0
        iz_prediction[iz_prediction >= n_epochs/2] = 1
        return self.binarize_stc(
            iz_prediction, self.fwd, smoothing_steps=self.smoothing_steps_final)
