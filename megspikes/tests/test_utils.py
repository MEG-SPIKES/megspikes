import os.path as op
from pathlib import Path

import numpy as np
import pytest
from megspikes.utils import (labels_to_mni, spike_snr_all_channels,
                             spike_snr_max_channel)


@pytest.fixture(scope="module", name='test_sample_path')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent
    sample_path = sample_path / 'tests_data' / 'test_utils'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.mark.regression
def test_snr_estimation(simulation_epochs_grad):
    data = simulation_epochs_grad['SRC1'].get_data()
    peak_ind = 500  # sample if sfreq = 1000 and epochs length = 1s
    n_max_channels = 20
    snr_all = spike_snr_all_channels(data, peak_ind)
    snr_max, max_ch = spike_snr_max_channel(data, peak_ind, n_max_channels)
    assert np.round(snr_all, 3) == 1.813
    assert np.round(snr_max, 3) == 6.207
    assert len(max_ch) == n_max_channels


def test_labels_to_mni(simulation):
    mni_resection, data, _ = labels_to_mni(
        simulation.labels, simulation.case_manager.fwd['ico5'],
        simulation.mne_subject, simulation.subjects_dir)
    assert sum(data != 0) == mni_resection.shape[0]
