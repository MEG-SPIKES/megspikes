import os.path as op
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
from megspikes.database.database import (Database, LoadDataset, SaveDataset,
                                         select_sensors)
from megspikes.simulation.simulation import Simulation


@pytest.fixture(name='simulation')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_database'
    sample_path.mkdir(exist_ok=True, parents=True)

    sim = Simulation(sample_path)
    sim.load_mne_dataset()
    sim.simulate_dataset(length=10)
    return sim


def test_empty_database():
    db = Database()
    ds = db.make_empty_dataset()
    assert type(ds) == xr.Dataset


def test_database_selection():
    db = Database()
    ds = db.make_empty_dataset()
    ds_grad = db.select_sensors(ds, 'grad', 'aspire_alphacsc_run_1')
    assert ds_grad.channel_names.shape[0] == 204


@pytest.mark.happy
def test_database(simulation):
    db = Database()
    db.read_case_info(simulation.case_manager.fif_file,
                      simulation.case_manager.fwd['ico5'])
    ds = db.make_empty_dataset()
    sens = 'grad'
    assert type(ds) == xr.Dataset
    ds_grad, _ = select_sensors(ds, sens, 'aspire_alphacsc_run_1')
    assert ds_grad.channel_names.shape[0] == 204
    ds.to_netcdf(simulation.case_manager.dataset)
    ds_grad['ica_components'].loc[:, :] = 4.
    ds_grad['ica_component_properties'].loc[
        dict(ica_component_property='mni_x')] = 6
    sdb = SaveDataset(simulation.case_manager.dataset,
                      sens, 'aspire_alphacsc_run_1')
    X = sdb.fit_transform((ds_grad, None))
    ldb = LoadDataset(simulation.case_manager.dataset,
                      sens, 'aspire_alphacsc_run_1')
    X = ldb.fit_transform(X)
    assert not np.isnan(X[0].ica_components.values).any()
    assert X[0].ica_components.values.all()
    assert X[0].ica_component_properties.values.any()

    ds_saved = xr.load_dataset(simulation.case_manager.dataset)
    assert not np.isnan(ds_saved.ica_components.values).any()
    assert not ds_saved.ica_components.values.all()

    sens = 'mag'
    ds_mag, _ = select_sensors(ds, sens, 'aspire_alphacsc_run_1')
    assert ds_mag.channel_names.shape[0] == 102
    ds_mag['ica_components'].loc[:, :] = 5.
    ds_mag['ica_component_properties'].loc[
        dict(ica_component_property='mni_x')] = 6
    sdb = SaveDataset(simulation.case_manager.dataset,
                      sens, 'aspire_alphacsc_run_1')
    X = sdb.fit_transform((ds_mag, None))
    ldb = LoadDataset(simulation.case_manager.dataset,
                      sens, 'aspire_alphacsc_run_1')
    X = ldb.fit_transform(X)
    assert not np.isnan(X[0].ica_components.values).any()
    assert X[0].ica_components.values.all()
    assert X[0].ica_component_properties.values.any()

    ds_saved = xr.load_dataset(simulation.case_manager.dataset)
    assert not np.isnan(ds_saved.ica_components.values).any()
    assert ds_saved.ica_components.values.all()
