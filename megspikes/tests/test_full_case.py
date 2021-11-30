import os.path as op
from pathlib import Path

import pytest
from megspikes.pipeline import (aspire_alphacsc_pipeline,
                                read_detection_iz_prediction_pipeline)
from megspikes.visualization.report import (report_detection,
                                            report_atoms_library)


@pytest.fixture(scope="module", name='test_sample_path')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent
    sample_path = sample_path / 'tests_data' / 'test_full_case'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.mark.happy
@pytest.mark.slow
@pytest.mark.parametrize("n_runs,runs,n_components", [
    (1, [0], 20), (2, [0, 1], 5)])
def test_full_case_analysis_pipeline(simulation_large, n_runs, runs,
                                     n_components):
    params = {
        'n_ica_components': n_components,
        'n_runs': n_runs,
        'runs': runs,
        'n_atoms': 2,  # FIXME: one atom cause bugs
        'PeakDetection': {'width': 2},
        'CleanDetections': {'n_cleaned_peaks': 50},
        'SelectAlphacscEvents': {
            'z_hat_threshold': 1.,
            'z_hat_threshold_min': 0.1}
    }
    pipe = aspire_alphacsc_pipeline(
        simulation_large.case_manager, update_params=params,
        rewrite_previous_results=True)
    dataset, raw = pipe.fit_transform(None)
    fname = f'test_report_detections_n_runs_{n_runs}_'\
            f'n_components_{n_components}.pdf'
    report_detections_path = simulation_large.case_manager.basic_folders[
                                 'REPORTS'] / fname
    fname = f'test_report_atoms_library_n_runs_{n_runs}_'\
            f'n_components_{n_components}.pdf'
    report_library_path = simulation_large.case_manager.basic_folders[
                      'REPORTS'] / fname
    report_detection(report_detections_path, dataset, raw)
    report_atoms_library(report_library_path, dataset, raw)

    params = {
        'PrepareClustersDataset': {'detection_sfreq': 200.}
    }
    pipe = read_detection_iz_prediction_pipeline(
        simulation_large.case_manager, params, True)
    _ = pipe.fit_transform((dataset, simulation_large.raw_simulation.copy()))
