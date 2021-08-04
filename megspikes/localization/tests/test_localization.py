import os.path as op
from pathlib import Path

import numpy as np
import pytest
from megspikes.localization.localization import (ClustersLocalization,
                                                 ICAComponentsLocalization,
                                                 PredictIZClusters)
from megspikes.utils import PrepareData
from megspikes.database.database import Database, select_sensors


@pytest.fixture(scope="module", name="test_sample_path")
def sample_path():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_localization'
    return sample_path


@pytest.fixture(scope="module", name="db")
def make_database():
    n_ica_comp = 3
    db = Database(times=np.linspace(0, 5, 5*200),
                  n_ica_components=n_ica_comp)
    return db


@pytest.mark.happy
def test_components_localization(dataset, simulation):
    sensors = 'grad'
    case = simulation.case_manager
    prep_data = PrepareData(sensors=sensors)
    ds_grad, sel = select_sensors(dataset, sensors, "aspire_alphacsc_run_1")
    (ds_grad, raw) = prep_data.fit_transform(
        (ds_grad, simulation.raw_simulation))
    cl = ICAComponentsLocalization(case=case, sensors=sensors)
    (ds_grad, raw) = cl.fit_transform((ds_grad, raw))


def test_clusters_localization(dataset, simulation):
    case = simulation.case_manager
    ds = dataset.copy(deep=True)
    prep_data = PrepareData(sensors=True)
    (_, raw) = prep_data.fit_transform((
        ds, simulation.raw_simulation))
    localizer = ClustersLocalization(
        case=case, db_name_detections='clusters_library_timestamps',
        db_name_clusters='clusters_library_cluster_id',
        detection_sfreq=200.)
    results = localizer.fit_transform((ds, raw))
    izpredictor = PredictIZClusters(case=case)
    results = izpredictor.fit_transform(results)
