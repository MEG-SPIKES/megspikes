import os.path as op
from pathlib import Path

import pytest
from megspikes.pipeline import aspike_alphacsc_pipeline


@pytest.fixture(scope="module", name='test_sample_path')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent
    sample_path = sample_path / 'tests_data' / 'test_pipeline'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.mark.pipeline
@pytest.mark.happy
@pytest.mark.slow
def test_pipeline(simulation, dataset):
    n_ica_components = len(dataset.ica_component)
    n_ica_peaks = 50
    resample = dataset.time.attrs['sfreq']
    n_atoms = len(dataset.alphacsc_atom)  # FIXME: one atom cause bugs
    z_hat_threshold = 1.
    z_hat_threshold_min = 0.1

    dataset.to_netcdf(simulation.case_manager.dataset)  # save empty dataset

    pipe = aspike_alphacsc_pipeline(
        simulation.case_manager, n_ica_components=n_ica_components,
        resample=resample, n_ica_peaks=n_ica_peaks, n_atoms=n_atoms,
        z_hat_threshold=z_hat_threshold,
        z_hat_threshold_min=z_hat_threshold_min,
        runs=dataset.run.values)
    _ = pipe.fit_transform(None)
