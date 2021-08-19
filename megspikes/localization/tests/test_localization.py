import os.path as op
from pathlib import Path

import mne
import numpy as np
import pytest
from megspikes.database.database import select_sensors
from megspikes.localization.localization import (
    AlphaCSCComponentsLocalization, ClustersLocalization,
    ICAComponentsLocalization, Localization, PeakLocalization,
    PredictIZClusters)
from megspikes.utils import PrepareData
from mne.beamformer import rap_music


@pytest.fixture(scope="module", name="test_sample_path")
def sample_path():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_localization'
    return sample_path


@pytest.mark.happy
@pytest.mark.parametrize(
    "sensors,spacing,n_ch,n_source",
    [('mag', 'oct5', 102, 1884),
     ('grad', 'oct5', 204, 1884),
     ('mag', 'ico5', 102, 18_840),
     ('grad', 'ico5', 204, 18_840),
     (True, 'oct5', 306, 1884),
     (True, 'ico5', 306, 18_840)])
def test_forward_model_setup(simulation, sensors, spacing, n_ch, n_source):
    case = simulation.case_manager
    loc = Localization()
    info, fwd, cov = loc.pick_sensors(case.info, case.fwd[spacing], sensors)
    assert info['nchan'] == n_ch
    assert fwd['nsource'] == n_source
    assert fwd['nchan'] == n_ch
    assert cov['dim'] == n_ch
    loc.setup_fwd(case, sensors, spacing=spacing)
    assert loc.info['nchan'] == n_ch
    assert loc.fwd['nsource'] == n_source
    assert loc.fwd['nchan'] == n_ch
    assert loc.cov['dim'] == n_ch


@pytest.mark.happy
def test_components_localization(aspire_alphacsc_random_dataset, simulation):
    dataset = aspire_alphacsc_random_dataset
    run = 0
    sensors = 'grad'
    case = simulation.case_manager
    prep_data = PrepareData(sensors=sensors)
    ds_grad, sel = select_sensors(dataset, sensors, run)
    (ds_grad, raw) = prep_data.fit_transform(
        (ds_grad, simulation.raw_simulation))
    cl = ICAComponentsLocalization(case=case, sensors=sensors)
    alphacl = AlphaCSCComponentsLocalization(case=case, sensors=sensors)
    (ds_grad, raw) = cl.fit_transform((ds_grad, raw))
    (ds_grad, raw) = alphacl.fit_transform((ds_grad, raw))


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
    assert np.linalg.norm(dip_pos[0] - mni_coords[0][::-1], ord=2) < 20


@pytest.mark.happy
def test_clusters_localization(clusters_random_dataset, simulation):
    dataset = clusters_random_dataset
    case = simulation.case_manager
    localizer = ClustersLocalization(case=case)
    results = localizer.fit_transform((dataset, simulation.raw_simulation))
    izpredictor = PredictIZClusters(case=case)
    results = izpredictor.fit_transform(results)
