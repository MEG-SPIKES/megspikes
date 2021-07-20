from typing import List, Tuple, Union
import logging

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
from ..utils import create_epochs, onset_slope_timepoints

import mne
mne.set_log_level("ERROR")


class Localization():
    def setup_fwd(self, case: CaseManager, sensors: Union[str, bool] = True):
        if not isinstance(case.fwd['oct5'], mne.Forward):
            raise RuntimeError("CaseManager don't include forward model")
        self.sensors = sensors
        self.case = case
        self.case_name = case.case
        self.info = case.info
        self.info = mne.pick_info(
            case.info, mne.pick_types(case.info, meg=self.sensors))
        self.n_channels = len(mne.pick_types(self.info, meg=True))

        self.fwd = case.fwd['oct5']
        if isinstance(self.sensors, str):
            self.fwd = mne.pick_types_forward(self.fwd, meg=self.sensors)

        self.bem = case.bem['oct5']
        self.trans = case.trans['oct5']
        self.freesurfer_dir = case.freesurfer_dir
        self.cov = mne.make_ad_hoc_cov(self.info)


class ComponentsLocalization(Localization, BaseEstimator, TransformerMixin):
    def __init__(self, case: CaseManager, sensors: Union[str, bool] = True):
        self.setup_fwd(case, sensors)

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None, **fit_params):
        components = X[0]['ica_components'].values[:, :self.n_channels].T
        evoked = mne.EvokedArray(components, self.info)
        dip = mne.fit_dipole(evoked, self.cov, self.bem, self.trans)[0]

        for n, d in enumerate(dip):
            pos_mni = mne.head_to_mni(
                d.pos[0],  self.case_name, self.fwd['mri_head_t'],
                subjects_dir=self.freesurfer_dir)
            X[0]['ica_components_localization'][n, :] = pos_mni
            X[0]['ica_components_gof'][n] = d.gof[0]
        return self

    def transform(self, X, **transform_params) -> Tuple[xr.Dataset,
                                                        mne.io.Raw]:
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
                 sfreq: int = 200, window: List[int] = [-20, 30]):
        self.setup_fwd(case, sensors)
        self.window = window
        self.sfreq = sfreq

    def transform(self, X, **transform_params) -> Tuple[xr.Dataset,
                                                        mne.io.Raw]:
        logging.info("ICA peaks are localized.")
        return X

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None, **fit_params):
        timestamps = X[0]["ica_peaks_timestamps"]
        timestamps = timestamps[timestamps != 0]
        n_peaks = len(timestamps)
        spikes = np.sort(timestamps)
        # NOTE: save sorted timestamps
        X[0]["ica_peaks_timestamps"][:n_peaks] = spikes

        data = X[1].get_data()
        window = (np.array(self.window)/1000)*self.sfreq
        mni_coords, subcorr = self.fast_music(
            data, self.info, spikes, window=window)
        X[0]["ica_peaks_localization"][:n_peaks, :] = mni_coords
        X[0]["ica_peaks_subcorr"][:n_peaks] = subcorr
        return self

    def fast_music(self, data: np.ndarray, info: mne.Info, spikes: np.ndarray,
                   window: List[int]):
        """
        :Authors:
        ------
            Daria Kleeva <dkleeva@gmail.com>
        """
        common_atr = self._prepare_rap_music_input(info, self.cov, self.fwd)
        zyx_mni = np.zeros((len(spikes), 3))
        subcorrs = np.zeros_like(spikes)

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
