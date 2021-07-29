import os.path as op
from pathlib import Path

import pytest

import numpy as np

from megspikes.casemanager.casemanager import CaseManager
from megspikes.localization.localization import (ICAComponentsLocalization,
                                                 ClustersLocalization,
                                                 PredictIZClusters)
from megspikes.simulation.simulation import Simulation
from megspikes.utils import PrepareData

import xarray as xr

sample_path = Path(op.dirname(__file__)).parent.parent.parent
sample_path = sample_path / 'example'
sample_path.mkdir(exist_ok=True)

sim = Simulation(sample_path)
sim.load_mne_dataset()
sim.simulate_dataset(length=5)

case = CaseManager(root=sample_path, case='sample',
                   free_surfer=sim.subjects_dir)
case.set_basic_folders()
case.select_fif_file(case.run)
case.prepare_forward_model()


@pytest.fixture(name="dataset")
def make_dataset():
    n_ica_comp = 3
    ica_components = xr.DataArray(
        np.random.sample((n_ica_comp, 204)),
        dims=("ica_component", "channels"))
    ica_components_localization = xr.DataArray(
        np.random.sample((n_ica_comp, 3)),
        dims=("ica_component", "mni_coord"))
    ica_components_gof = xr.DataArray(
        np.random.sample(n_ica_comp),
        dims=("ica_component"))
    clusters_library_timestamps = xr.DataArray(
        np.array([120., 160., 250., 310.]),
        dims=("alpha_detections"))
    clusters_library_cluster_id = xr.DataArray(
        np.array([0.0, 1.0, 1.0, 0.0]),
        dims=("alpha_detections"))
    iz_predictions = xr.DataArray(
        np.zeros((4, sum(
            [len(h['vertno']) for h in case.fwd['ico5']['src']]))),
        dims=("prediction_type", "fwd_source"),
        coords={
            "prediction_type": ['alphacsc_peak', 'alphacsc_slope',
                                'manual', 'resection']
            },
        name="iz_predictions")

    ds = xr.Dataset(data_vars={
        "ica_components": ica_components,
        "ica_components_localization": ica_components_localization,
        "ica_components_gof": ica_components_gof,
        "clusters_library_timestamps": clusters_library_timestamps,
        "clusters_library_cluster_id": clusters_library_cluster_id,
        "iz_predictions": iz_predictions})
    return ds


def test_components_localization(dataset):
    ds = dataset.copy(deep=True)
    case.prepare_forward_model()
    prep_data = PrepareData(sensors='grad')
    (_, raw) = prep_data.fit_transform((ds, sim.raw_simulation))
    cl = ICAComponentsLocalization(case=case, sensors='grad')
    results = cl.fit_transform((ds, raw))
    assert results[0]['ica_components_localization'].any()
    assert results[0]['ica_components_gof'].any()
    del results, ds

    ds = dataset.copy(deep=True)
    case.prepare_forward_model(sensors='grad')
    prep_data = PrepareData(sensors='grad')
    (_, raw) = prep_data.fit_transform((ds, sim.raw_simulation))
    cl = ICAComponentsLocalization(case=case)
    results = cl.fit_transform((ds, raw))
    assert results[0]['ica_components_localization'].any()
    assert results[0]['ica_components_gof'].any()


def test_clusters_localization(dataset):
    ds = dataset.copy(deep=True)
    prep_data = PrepareData(sensors=True)
    (_, raw) = prep_data.fit_transform((
        ds, sim.raw_simulation))
    localizer = ClustersLocalization(
        case=case, db_name_detections='clusters_library_timestamps',
        db_name_clusters='clusters_library_cluster_id',
        detection_sfreq=200.)
    results = localizer.fit_transform((ds, raw))
    izpredictor = PredictIZClusters(case=case)
    results = izpredictor.fit_transform(results)