import os.path as op
from pathlib import Path

import pytest
from megspikes.pipeline import (aspire_alphacsc_pipeline,
                                iz_prediction_pipeline, manual_pipeline)


@pytest.fixture(scope="module", name='test_sample_path')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent
    sample_path = sample_path / 'tests_data' / 'test_pipeline'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.mark.happy
@pytest.mark.slow
def test_aspire_alphacsc_pipeline(simulation):
    n_ica_components = 5
    n_ica_peaks = 50
    resample = 200.
    n_atoms = 2  # FIXME: one atom cause bugs
    z_hat_threshold = 1.
    z_hat_threshold_min = 0.1
    runs = [0, 1]

    pipe = aspire_alphacsc_pipeline(
        simulation.case_manager, n_ica_components=n_ica_components,
        resample=resample, n_ica_peaks=n_ica_peaks, n_atoms=n_atoms,
        z_hat_threshold=z_hat_threshold,
        z_hat_threshold_min=z_hat_threshold_min,
        runs=runs)
    _ = pipe.fit_transform(None)


@pytest.mark.happy
def test_iz_prediction_pipeline(simulation):
    pipe = iz_prediction_pipeline(simulation.case_manager, 1000.)
    atoms_lib = {'spikes': simulation.detections}
    raw = simulation.raw_simulation.copy()
    _ = pipe.fit_transform((atoms_lib, raw))


@pytest.mark.happy
@pytest.mark.slow
def test_manual_pipeline(simulation):
    pipe = manual_pipeline(simulation.case_manager, simulation.detections,
                           simulation.clusters - 1)
    _ = pipe.fit_transform((None, simulation.raw_simulation.copy()))
