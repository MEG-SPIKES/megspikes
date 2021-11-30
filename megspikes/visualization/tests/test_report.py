import os.path as op
from pathlib import Path

import pytest
from megspikes.visualization.report import (report_detection,
                                            report_atoms_library)


@pytest.fixture(scope="module", name="test_sample_path")
def sample_path2():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_report'
    return sample_path


@pytest.mark.happy
def test_detection_report(simulation, aspire_alphacsc_random_dataset):
    ds = aspire_alphacsc_random_dataset.copy(deep=True)
    report_path = simulation.case_manager.basic_folders[
                      'REPORTS'] / 'test_report_detections.pdf'
    report_detection(report_path, ds, simulation.raw_simulation.copy())

@pytest.mark.happy
def test_atom_library_report(simulation, aspire_alphacsc_random_dataset):
    ds = aspire_alphacsc_random_dataset.copy(deep=True)
    report_path = simulation.case_manager.basic_folders[
                      'REPORTS'] / 'test_report_atoms_library.pdf'
    report_atoms_library(report_path, ds, simulation.raw_simulation.copy())


@pytest.mark.happy
def test_cluster_report(simulation, clusters_random_dataset):
    pass
