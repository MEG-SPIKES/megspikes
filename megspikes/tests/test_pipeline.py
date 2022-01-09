import os.path as op
from pathlib import Path

import numpy as np
import pytest
from megspikes.pipeline import (aspire_alphacsc_pipeline,
                                iz_prediction_pipeline,
                                read_detection_iz_prediction_pipeline,
                                update_default_params)
from megspikes.visualization.report import report_detection, \
    report_atoms_library


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
        'PrepareData': {'alpha_notch': 10},
        'PeakDetection': {'width': 2},
        'CleanDetections': {'n_cleaned_peaks': 50},
        'SelectAlphacscEvents': {
            'z_hat_threshold': 1.,
            'z_hat_threshold_min': 0.1}
    }
    pipe = aspire_alphacsc_pipeline(
        simulation.case_manager, update_params=params,
        rewrite_previous_results=True,
        manual_ica_components={'grad': (None, (0, 1,)), 'mag': ((0,), None)})
    dataset, raw = pipe.fit_transform(None)
    fname = f'test_report_detections.pdf'
    report_detections_path = simulation.case_manager.basic_folders[
                                 'REPORTS'] / fname
    fname = f'test_report_atoms_library.pdf'
    report_library_path = simulation.case_manager.basic_folders[
                              'REPORTS'] / fname
    report_detection(report_detections_path, dataset, raw)
    report_atoms_library(report_library_path, dataset, raw)


@pytest.mark.happy
def test_iz_prediction_pipeline(simulation):
    params = {
        'PrepareClustersDataset': {'detection_sfreq': 1000.}
    }
    pipe = iz_prediction_pipeline(
        simulation.case_manager, params, rewrite_previous_results=True)
    atoms_lib = {'spikes': simulation.detections}
    raw = simulation.raw_simulation.copy()
    _ = pipe.fit_transform((atoms_lib, raw))


def test_iz_prediction_pipeline_mne_dataset(mne_example_dataset):
    """Test with the real data file"""
    params = {
        'PrepareClustersDataset': {'detection_sfreq': 1000.}
    }
    case = mne_example_dataset.case_manager
    pipe = iz_prediction_pipeline(case, params, rewrite_previous_results=True)
    atoms_lib = {'spikes': np.int32(np.arange(1, 10) * 1000)}
    raw = mne_example_dataset.raw_simulation.copy()
    _ = pipe.fit_transform((atoms_lib, raw))


@pytest.mark.happy
@pytest.mark.slow
def test_read_results_iz_prediction_pipeline(simulation,
                                             aspire_alphacsc_random_dataset):
    params = {
        'PrepareClustersDataset': {'detection_sfreq': 200.}
    }
    pipe = read_detection_iz_prediction_pipeline(
        simulation.case_manager, params, rewrite_previous_results=True)
    _ = pipe.fit_transform((aspire_alphacsc_random_dataset,
                            simulation.raw_simulation.copy()))


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
