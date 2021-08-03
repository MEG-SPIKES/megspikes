import os.path as op
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from megspikes.localization.localization import (ClustersLocalization,
                                                 ICAComponentsLocalization,
                                                 PredictIZClusters)
from megspikes.simulation.simulation import Simulation
from megspikes.utils import PrepareData
from megspikes.database.database import Database, select_sensors


@pytest.fixture(name='simulation')
def run_simulation():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_localization'
    sample_path.mkdir(exist_ok=True, parents=True)

    sim = Simulation(sample_path)
    sim.load_mne_dataset()
    sim.simulate_dataset(length=5)
    return sim


@pytest.fixture(name="dataset")
def make_dataset(simulation):
    # case = simulation.case_manager
    n_ica_comp = 3
    db = Database(times=np.linspace(0, 5, 5*200),
                  n_ica_components=n_ica_comp)
    ds = db.make_empty_dataset()

    ds.ica_components.values = np.random.sample(ds.ica_components.shape)
    ds.ica_component_properties.values = np.random.sample(
        ds.ica_component_properties.shape)

    # ica_components_localization = xr.DataArray(
    #     np.random.sample((n_ica_comp, 3)),
    #     dims=("ica_component", "mni_coord"))
    # ica_components_gof = xr.DataArray(
    #     np.random.sample(n_ica_comp),
    #     dims=("ica_component"))
    # clusters_library_timestamps = xr.DataArray(
    #     np.array([120., 160., 250., 310.]),
    #     dims=("alpha_detections"))
    # clusters_library_cluster_id = xr.DataArray(
    #     np.array([0.0, 1.0, 1.0, 0.0]),
    #     dims=("alpha_detections"))
    # iz_predictions = xr.DataArray(
    #     np.zeros((4, sum(
    #         [len(h['vertno']) for h in case.fwd['ico5']['src']]))),
    #     dims=("prediction_type", "fwd_source"),
    #     coords={
    #         "prediction_type": ['alphacsc_peak', 'alphacsc_slope',
    #                             'manual', 'resection']},
    #     name="iz_predictions")
    # ds = xr.Dataset(data_vars={
    #     "ica_components": ica_components,
    #     "ica_components_localization": ica_components_localization,
    #     "ica_components_gof": ica_components_gof,
    #     "clusters_library_timestamps": clusters_library_timestamps,
    #     "clusters_library_cluster_id": clusters_library_cluster_id,
    #     "iz_predictions": iz_predictions})
    return ds


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
