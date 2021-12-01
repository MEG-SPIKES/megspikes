import os.path as op
from pathlib import Path

import mne
import numpy as np
import pytest
from megspikes.utils import (PrepareData, labels_to_mni, prepare_data,
                             spike_snr_all_channels, spike_snr_max_channel,
                             brainstorm_events_export, read_dip)

mne.set_log_level("ERROR")


@pytest.fixture(scope="module", name='test_sample_path')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent
    sample_path = sample_path / 'tests_data' / 'test_utils'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.mark.parametrize("meg", [True, 'grad', 'mag'])
@pytest.mark.parametrize(
    "resample,filtering,alpha_notch",
    [(1000., [2, 90, 50], None),
     (200., [2, 90, 50], 12)])
def test_prepare_data(simulation, meg, resample, filtering, alpha_notch):
    raw = simulation.raw_simulation.copy()
    raw = prepare_data(raw, meg=meg, filtering=filtering, resample=resample,
                       alpha_notch=alpha_notch)
    assert raw.info['sfreq'] == resample
    assert raw.info['highpass'] == filtering[0]
    assert raw.info['lowpass'] == filtering[1]
    if isinstance(meg, bool):
        assert raw.info['nchan'] == 306
    elif meg == 'grad':
        assert raw.info['nchan'] == 204
    else:
        assert raw.info['nchan'] == 102


@pytest.mark.parametrize("meg", [True, 'grad', 'mag'])
@pytest.mark.parametrize(
    "resample,filtering,alpha_notch",
    [(1000., [2, 90, 50], None),
     (200., [2, 90, 50], 12),
     (None, None, None)])
def test_prepare_data_transformer(simulation, meg, resample, filtering,
                                  alpha_notch):
    pd = PrepareData(data_file=simulation.case_manager.fif_file, sensors=meg,
                     filtering=filtering, resample=resample,
                     alpha_notch=alpha_notch)
    raw = simulation.raw_simulation.copy()
    raw = pd.fit_transform(raw)
    if resample is None:
        raw.info['sfreq'] == simulation.raw_simulation.info['sfreq']
    else:
        assert raw.info['sfreq'] == resample
    if filtering is None:
        assert raw.info['highpass'] == \
            simulation.raw_simulation.info['highpass']
        assert raw.info['lowpass'] == simulation.raw_simulation.info['lowpass']
    else:
        assert raw.info['highpass'] == filtering[0]
        assert raw.info['lowpass'] == filtering[1]
    if isinstance(meg, bool):
        assert raw.info['nchan'] == 306
    elif meg == 'grad':
        assert raw.info['nchan'] == 204
    else:
        assert raw.info['nchan'] == 102


@pytest.mark.regression
def test_snr_estimation(simulation_epochs_grad):
    data = simulation_epochs_grad['SRC1'].get_data()
    peak_ind = 500  # sample if sfreq = 1000 and epochs length = 1s
    n_max_channels = 20
    snr_all = spike_snr_all_channels(data, peak_ind)
    snr_max, max_ch = spike_snr_max_channel(data, peak_ind, n_max_channels)
    assert np.isclose(snr_all, 1.8, atol=0.5)
    assert np.isclose(snr_max, 6.2, atol=0.5)
    assert len(max_ch) == n_max_channels


def test_labels_to_mni(simulation):
    mni_resection, data, _ = labels_to_mni(
        simulation.labels, simulation.case_manager.fwd['ico5'],
        simulation.mne_subject, simulation.subjects_dir)
    assert sum(data != 0) == mni_resection.shape[0]


@pytest.mark.happy
def test_brainstorm_export(test_sample_path):
    save_path = test_sample_path / 'bst_events.mat'
    times = {f'{i}_marker': np.random.random(100) for i in range(4)}
    brainstorm_events_export(save_path, times)


@pytest.mark.happy
def test_reading_dip_files(test_sample_path):
    root = Path(op.dirname(__file__))
    dip_path = root / 'data' / 'test_dipoles.dip'
    dips = read_dip(dip_path)
    assert dips[0] == 555555
