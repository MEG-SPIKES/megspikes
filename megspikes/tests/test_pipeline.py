import os.path as op
from pathlib import Path

import pytest
from megspikes.pipeline import (aspire_alphacsc_pipeline,
                                iz_prediction_pipeline, manual_pipeline,
                                update_default_params)


@pytest.fixture(scope="module", name='test_sample_path')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent
    sample_path = sample_path / 'tests_data' / 'test_pipeline'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.mark.happy
@pytest.mark.slow
def test_aspire_alphacsc_pipeline(simulation):
    params = {
        'n_ica_components': 5,
        'n_runs': 2,
        'runs': [0, 1],
        'n_atoms': 2,  # FIXME: one atom cause bugs
        'PeakDetection': {'width': 2},
        'CleanDetections': {'n_cleaned_peaks': 50},
        'SelectAlphacscEvents': {
            'z_hat_threshold': 1.,
            'z_hat_threshold_min': 0.1}
    }
    pipe = aspire_alphacsc_pipeline(
        simulation.case_manager, update_params=params)
    _ = pipe.fit_transform(None)


@pytest.mark.happy
def test_iz_prediction_pipeline(simulation):
    params = {
        'PrepareClustersDataset': {'detection_sfreq': 1000.}
    }
    pipe = iz_prediction_pipeline(simulation.case_manager, params)
    atoms_lib = {'spikes': simulation.detections}
    raw = simulation.raw_simulation.copy()
    _ = pipe.fit_transform((atoms_lib, raw))


@pytest.mark.happy
@pytest.mark.slow
def test_manual_pipeline(simulation):
    pipe = manual_pipeline(simulation.case_manager, simulation.detections,
                           simulation.clusters - 1)
    _ = pipe.fit_transform((None, simulation.raw_simulation.copy()))


def test_update_default_params():
    params = {
        'param_global': 1,
        'param_global2': 1,
        'param_local': {
            'b': 5,
            'param_global': 1},
        'param_local_local': {
            'param_local1': {
                'c': 6,
                'param_global': 1}}
    }

    updates = {
        'param_global': 2,
        'param_local': {
            'b': 6},
        'param_local_local': {
            'param_local1': {
                'c': 0}}
    }

    result = {
        'param_global': 2,
        'param_global2': 1,
        'param_local': {
            'b': 6,
            'param_global': 2},
        'param_local_local': {
            'param_local1': {
                'c': 0,
                'param_global': 2}}
    }

    updated_params = update_default_params(params, updates)
    assert updated_params == result
