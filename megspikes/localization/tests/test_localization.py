import os.path as op
from pathlib import Path

import numpy as np
import pytest
from mne.beamformer import rap_music
import mne
from megspikes.database.database import select_sensors
from megspikes.localization.localization import (ClustersLocalization,
                                                 ICAComponentsLocalization,
                                                 PeakLocalization,
                                                 PredictIZClusters)
from megspikes.utils import PrepareData


@pytest.fixture(scope="module", name="test_sample_path")
def sample_path():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_localization'
    return sample_path


@pytest.mark.happy
def test_components_localization(dataset, simulation):
    pipeline = "aspire_alphacsc_run_1"
    sensors = 'grad'
    case = simulation.case_manager
    prep_data = PrepareData(sensors=sensors)
    ds_grad, sel = select_sensors(dataset, sensors, pipeline)
    (ds_grad, raw) = prep_data.fit_transform(
        (ds_grad, simulation.raw_simulation))
    cl = ICAComponentsLocalization(case=case, sensors=sensors)
    (ds_grad, raw) = cl.fit_transform((ds_grad, raw))


@pytest.mark.xfail
def test_fast_rap_music(simulation):
    sensors = 'grad'
    case = simulation.case_manager
    pk = PeakLocalization(case=case, sensors=sensors)
    pd = PrepareData(data_file=simulation.case_manager.fif_file,
                     sensors=sensors, resample=200.)
    _, raw = pd.fit_transform(None)
    data = raw.get_data()
    timestamps = np.array([100, 200, 300], dtype=np.int32)
    mni_coords, subcorr = pk.fast_music(
        data, raw.info, timestamps, window=[-4, 6])  # samples, sfreq=200Hz
    for time, fast_rap in zip(timestamps, mni_coords):
        fast_rap = fast_rap[::-1]
        evoked = mne.EvokedArray(data[:, time-4:time+6], raw.info)
        dipoles = rap_music(evoked, pk.fwd, pk.cov, n_dipoles=5)
        mni_pos = mne.head_to_mni(
            dipoles[0].pos,  pk.case_name, pk.fwd['mri_head_t'],
            subjects_dir=pk.freesurfer_dir)
        # less than 20mm
        assert np.linalg.norm(mni_pos[0] - fast_rap, ord=2) < 20


@pytest.mark.xfail
def test_fast_rap_music_details(simulation):
    from mne.datasets import sample
    data_path = sample.data_path()
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

    # Read the evoked response and crop it
    condition = 'Right Auditory'
    evoked = mne.read_evokeds(evoked_fname, condition=condition,
                              baseline=(None, 0))
    # select N100
    evoked.crop(tmin=0.05, tmax=0.15)

    evoked.pick_types(meg='grad', eeg=False)

    cov = mne.make_ad_hoc_cov(evoked.info)

    # Read the forward solution
    forward = mne.read_forward_solution(fwd_fname)

    case = simulation.case_manager
    pk = PeakLocalization(case=case, sensors='grad')
    pk.fwd = forward
    pk.cov = cov
    mni_coords, subcorr = pk.fast_music(
        evoked.data, evoked.info, np.array([500]), window=[-50, 150])
    dipoles = rap_music(evoked, forward, cov, n_dipoles=5)
    dip_pos = mne.head_to_mni(
            dipoles[0].pos,  pk.case_name, pk.fwd['mri_head_t'],
            subjects_dir=pk.freesurfer_dir)
    np.linalg.norm(dip_pos[0] - mni_coords[0][::-1], ord=2) < 20


def test_clusters_localization(dataset, simulation):
    case = simulation.case_manager
    ds = dataset.copy(deep=True)
    prep_data = PrepareData(sensors=True)
    (_, raw) = prep_data.fit_transform((
        ds, simulation.raw_simulation))
    localizer = ClustersLocalization(
        case=case, db_name_detections='clusters_library_timestamps',
        db_name_clusters='clusters_library_cluster_id',
        detection_sfreq=200.)
    results = localizer.fit_transform((ds, raw))
    izpredictor = PredictIZClusters(case=case)
    results = izpredictor.fit_transform(results)
