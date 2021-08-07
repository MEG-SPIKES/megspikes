import logging
from typing import Any, List, Tuple, Union

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


class Localization():
    def setup_fwd(self, case: CaseManager, sensors: Union[str, bool] = True,
                  spacing: str = 'oct5'):
        if not isinstance(case.fwd[spacing], mne.Forward):
            raise RuntimeError("CaseManager don't include forward model")
        self.sensors = sensors
        self.case = case
        self.case_name = case.case
        self.info = case.info
        self.info = mne.pick_info(
            case.info, mne.pick_types(case.info, meg=self.sensors))
        self.n_channels = len(mne.pick_types(self.info, meg=True))

        self.fwd = case.fwd[spacing]
        if isinstance(self.sensors, str):
            self.fwd = mne.pick_types_forward(self.fwd, meg=self.sensors)

        self.bem = case.bem[spacing]
        self.trans = case.trans[spacing]
        self.freesurfer_dir = case.freesurfer_dir
        self.cov = mne.make_ad_hoc_cov(self.info)

    def make_labels_ts(self, stc: mne.SourceEstimate,
                       inverse_operator: mne.minimum_norm.InverseOperator,
                       mode: str = 'mean'):
        """Extract labels (anatomical) time courses

        Parameters
        ----------
        stc : mne.SourceEstimate
            [description]

        Returns
        -------
        label_tc
            array
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
                     min_sources: int = 10) -> np.ndarray:
        """Binarization and smoothing of one SourceEstimate timepoint and
        converting to mne.SourceEstimate.

        Parameters
        ----------
        data : np.ndarray
            1d, length is equal the number of fwd sources
        smoothing_steps : int, optional
            smoothing individual spikes and final results, by default 3

        """
        vertices = [i['vertno'] for i in fwd['src']]

        # normalize data
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
            stc, 0, smoothing_steps=smoothing_steps)
        final_data[:len(stc.vertices[0])] = lh_data

        # Smooth final surface right hemi
        rh_data = self._smooth_binarized_stc(
            stc, 1, smoothing_steps=smoothing_steps)
        final_data[len(stc.vertices[0]):] = rh_data
        final_data[final_data > 0] = 1

        # return binary map
        return final_data

    def _smooth_binarized_stc(self, stc: mne.SourceEstimate, hemi_idx: int,
                              smoothing_steps: int = 10):
        """Smooth binary SourceEstimate """
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

    def array_to_stc(self, data: np.ndarray) -> mne.SourceEstimate:
        """Convert SourceEstimate data to mne.SourceEstimate object"""
        vertices = [i['vertno'] for i in self.fwd['src']]
        return mne.SourceEstimate(
            data, vertices, tmin=0, tstep=0.001, subject=self.case_name)


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
        downsample freq, by default 1000.
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
            X[0], 'detection_properties', full_coords[:, 2],
            dict(detection_property='mni_x'))
        check_and_write_to_dataset(
            X[0], 'detection_properties', full_coords[:, 1],
            dict(detection_property='mni_y'))
        check_and_write_to_dataset(
            X[0], 'detection_properties', full_coords[:, 0],
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
                 db_name_detections: str,
                 db_name_clusters: str,
                 detection_sfreq: float = 200.,
                 sensors: Union[str, bool] = True,
                 inv_method: str = 'MNE',
                 epochs_window: Tuple[float] = (-0.5, 0.5)):
        self.setup_fwd(case, sensors)
        self.detection_sfreq = detection_sfreq
        self.db_name_detections = db_name_detections
        self.db_name_clusters = db_name_clusters
        self.inv_method = inv_method
        self.epochs_window = epochs_window

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        self.inverse_operator = make_inverse_operator(
            self.info, self.fwd, self.cov, depth=None, fixed=False)
        timestamps = X[0][self.db_name_detections].values
        clusters = X[0][self.db_name_clusters].values
        non_zero = timestamps != 0
        timestamps = timestamps[non_zero]
        clusters = clusters[non_zero]

        # resample timestamps
        timestamps = self.resample_timestamps(
            timestamps, self.detection_sfreq, X[1].info['sfreq'])

        epochs_width = (np.abs(self.epochs_window[0]) +
                        np.abs(self.epochs_window[1])) * X[1].info['sfreq']
        epochs_width = np.int32(epochs_width)
        epochs_times = np.linspace(
                    self.epochs_window[0], self.epochs_window[1], epochs_width)
        channels = mne.pick_types(X[1].info, meg=True)
        n_clusters = len(np.unique(clusters))
        all_clusters = np.int32(np.unique(clusters))
        array_shape = (n_clusters, len(channels), len(epochs_times))
        clusters_lib_evoked = xr.DataArray(
            np.zeros(array_shape),
            dims=("cluster_id", "meg_channels", "cluster_lib_times"),
            coords={
                "cluster_id": all_clusters,
                "meg_channels": channels,
                "cluster_lib_times": epochs_times},
            name="clusters_lib_evoked")

        fwd_vertno = np.hstack([h['vertno'] for h in self.fwd['src']])
        array_shape = (n_clusters, len(fwd_vertno), len(epochs_times))
        clusters_lib_sources = xr.DataArray(
            np.zeros(array_shape),
            dims=("cluster_id", "fwd_vertno", "cluster_lib_times"),
            coords={
                "cluster_id": all_clusters,
                "fwd_vertno": fwd_vertno,
                "cluster_lib_times": epochs_times},
            name="clusters_lib_sources")

        clusters_lib_slope_timepoints = xr.DataArray(
            np.zeros((n_clusters, 3)),
            dims=("cluster_id", "spike_slope_timepoints"),
            coords={
                "cluster_id": all_clusters,
                "spike_slope_timepoints": ['baseline', 'slope', 'peak']
                },
            name="clusters_lib_slope_timepoints")

        for cluster in all_clusters:
            # select timestamps for the cluster
            times = timestamps[clusters == cluster]
            # add first sample
            times += X[1].first_samp
            # Create epochs for the cluster
            epochs = create_epochs(
                X[1], times, tmin=self.epochs_window[0],
                tmax=self.epochs_window[1])
            # Create Evoked
            evoked = epochs.average()
            clusters_lib_evoked.loc[
                cluster, :, :] = evoked.data[:, :epochs_width]

            # minimum norm
            stc, label_ts = self.minimum_norm(evoked)

            clusters_lib_sources.loc[
                cluster, :, :] = stc.data[:, :epochs_width]

            # Slope components
            # t1 - slope (20%), t2 - slope (50%), t3 - peak
            slope_times = onset_slope_timepoints(
                label_ts[0].mean(axis=0))

            # Update final atoms table for ROC and AtomViewer
            clusters_lib_slope_timepoints.loc[
                cluster, :] = slope_times

        X[0]['clusters_lib_evoked'] = clusters_lib_evoked
        X[0]['clusters_lib_sources'] = clusters_lib_sources
        X[0]['clusters_lib_slope_timepoints'] = clusters_lib_slope_timepoints
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        logging.info("Clusters are localized.")
        return X

    def minimum_norm(self, evoked: Union[mne.Evoked, mne.EvokedArray],
                     inv_method: str = 'MNE') -> Tuple[
                         mne.SourceEstimate, np.ndarray]:
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        stc = apply_inverse(
            evoked, self.inverse_operator, lambda2, inv_method, pick_ori=None)
        label_ts = self.make_labels_ts(stc, self.inverse_operator)
        return stc, label_ts

    def resample_timestamps(self, timestamps: np.ndarray,
                            from_sfreq: float = 200.,
                            to_sfreq: float = 1000.) -> np.ndarray:
        return (timestamps / from_sfreq) * to_sfreq


class PredictIZClusters(Localization, BaseEstimator, TransformerMixin):
    def __init__(self, case: CaseManager,
                 sensors: Union[str, bool] = True,
                 smoothing_steps_one_cluster: int = 3,
                 smoothing_steps_final: int = 10,
                 amplitude_threshold: float = 0.5,
                 min_sources: int = 10):
        self.setup_fwd(case, sensors)
        self.smoothing_steps_one_cluster = smoothing_steps_one_cluster
        self.smoothing_steps_final = smoothing_steps_final
        self.amplitude_threshold = amplitude_threshold
        self.min_sources = min_sources

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        n_clusters = len(X[0].cluster_id)
        for time in ['peak', 'slope']:
            clusters_stcs = []
            for cluster in X[0].cluster_id.values:
                # Select slope timepoint from database
                slope_time = X[0]['clusters_lib_slope_timepoints'].loc[
                    cluster, time].values
                slope_time = np.int64(slope_time)
                # Select SourceEstimate data from Datbase
                stc_cluster = X[0][
                    'clusters_lib_sources'].loc[cluster].values[:, slope_time]
                # Binarize SourceEstimate
                stc_cluster_bin = self.binarize_stc(
                    stc_cluster, self.fwd, self.smoothing_steps_one_cluster,
                    self.amplitude_threshold, self.min_sources)
                clusters_stcs.append(stc_cluster_bin)

            # Binarize stc again
            iz_prediciton = np.stack(clusters_stcs, axis=-1).sum(axis=-1)
            iz_prediciton[iz_prediciton < n_clusters/2] = 0
            iz_prediciton[iz_prediciton >= n_clusters/2] = 1
            iz_prediciton = self.binarize_stc(
                iz_prediciton, self.fwd, self.smoothing_steps_final,
                self.amplitude_threshold, self.min_sources)
            X[0]["iz_predictions"].loc[f'alphacsc_{time}', :] = iz_prediciton
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        logging.info("Irritative zone prediction using clusters is finished.")
        return X
