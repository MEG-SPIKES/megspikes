import os.path as op
from pathlib import Path

import pytest
from megspikes.database.database import (check_and_read_from_dataset,
                                         check_and_write_to_dataset)
from megspikes.visualization.visualization import (ClusterSlopeViewer,
                                                   DetectionsViewer)


@pytest.fixture(scope="module", name="test_sample_path")
def sample_path2():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_visualization'
    return sample_path


@pytest.mark.happy
def test_detections_viewer(simulation, aspire_alphacsc_random_dataset):
    ds = aspire_alphacsc_random_dataset.copy(deep=True)
    atoms = check_and_read_from_dataset(
        ds, 'detection_properties',
        dict(detection_property='alphacsc_atom'))
    atoms[:, :, 1000] = 0
    atoms[:, :, 1500] = 0
    check_and_write_to_dataset(
        ds, 'detection_properties', atoms,
        dict(detection_property='alphacsc_atom'))
    pp = DetectionsViewer(ds, simulation.case_manager)
    ica_app = pp.view_ica()
    del ica_app

    ica_source_app = pp.view_ica_sources_and_peaks()
    del ica_source_app

    ica_peaks_localization_app = pp.view_ica_peak_localizations()
    del ica_peaks_localization_app

    atoms_app = pp.view_alphacsc_atoms()
    del atoms_app

    atoms_clusters_app = pp.view_alphacsc_clusters(
        simulation.raw_simulation.copy())
    del atoms_clusters_app


@pytest.mark.happy
def test_cluster_slope_viewer(simulation, clusters_random_dataset):
    pc = ClusterSlopeViewer(clusters_random_dataset,
                            simulation.case_manager)
    app = pc.view()
    del app
