from typing import List, Tuple, Union

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


class ComponentsLocalization(BaseEstimator, TransformerMixin):
    def __init__(self, case: CaseManager):
        if not isinstance(case.fwd['oct5'], mne.Forward):
            raise RuntimeError("CaseManager don't include forward model")
        self.case_name = case.case
        self.info = case.info
        self.fwd = case.fwd['oct5']
        self.bem = case.bem['oct5']
        self.trans = case.trans['oct5']
        self.freesurfer_dir = case.freesurfer_dir
        self.cov = mne.make_ad_hoc_cov(self.info)

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        components = X[0]['ica_components'].values.T
        evoked = mne.EvokedArray(components, self.info)
        dip = mne.fit_dipole(evoked, self.cov, self.bem, self.trans)[0]

        for n, d in enumerate(dip):
            pos_mni = mne.head_to_mni(
                d.pos[0],  self.case_name, self.fwd['mri_head_t'],
                subjects_dir=self.freesurfer_dir)
            X[0]['ica_components_localization'][n, :] = pos_mni
            X[0]['ica_components_gof'][n] = d.gof[0]
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        return X
