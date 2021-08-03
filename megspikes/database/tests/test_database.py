import os.path as op
from pathlib import Path

import pytest
import xarray as xr
from megspikes.database.database import Database
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


def test_database_read_info(simulation):
    db = Database()
    db.read_case_info(simulation.case_manager.fif_file,
                      simulation.case_manager.fwd['ico5'])
    ds = db.make_empty_dataset()
    assert type(ds) == xr.Dataset
    ds_grad = db.select_sensors(ds, 'grad', 'aspire_alphacsc_run_1')
    assert ds_grad.channel_names.shape[0] == 204
