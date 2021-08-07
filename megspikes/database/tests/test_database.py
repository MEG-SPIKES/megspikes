import os.path as op
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from megspikes.database.database import (Database, LoadDataset, SaveDataset,
                                         check_and_read_from_dataset,
                                         check_and_write_to_dataset,
                                         read_meg_info_for_database,
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


@pytest.mark.happy
def test_make_datasets():
    db = Database(
        sensors=['grad', 'mag'],
        channel_names=[f'MEG {i}' for i in range(306)],
        channels_by_sensors={'grad': np.arange(0, 204),
                             'mag': np.arange(204, 306)},
        fwd_sources=[[i for i in range(1, 10_000)],
                     [i for i in range(10_001, 20_000)]])
    ds = db.make_aspire_alphacsc_dataset(
        times=np.linspace(0, 10, 10_000),
        n_ica_components=2,
        n_atoms=2,
        atom_length=0.5,
        n_runs=4,
        sfreq=200.)
    assert type(ds) == xr.Dataset
    ds_grad = db.select_sensors(ds, 'grad', 0)
    assert ds_grad.channel_names.shape[0] == 204
    ds = db.make_clusters_dataset(
        times=np.linspace(0, 10, 10_000),
        n_clusters=5,
        evoked_length=1.,
        sfreq=1000.)
    assert type(ds) == xr.Dataset


@pytest.mark.happy
def test_database(simulation):
    case = simulation.case_manager
    db = read_meg_info_for_database(
        simulation.case_manager.fif_file, case.fwd['ico5'])
    ds = db.make_aspire_alphacsc_dataset(
        times=np.linspace(0, 10, 10_000),
        n_ica_components=2,
        n_atoms=2,
        atom_length=0.5,
        n_runs=4,
        sfreq=200.)

    sens = 'grad'
    run = 0
    ds_grad = db.select_sensors(ds, sens, run)
    ds.to_netcdf(simulation.case_manager.dataset)
    ds_grad['ica_components'].loc[:, :] = 4.
    ds_grad['ica_component_properties'].loc[
        dict(ica_component_property='mni_x')] = 6
    sdb = SaveDataset(simulation.case_manager.dataset,
                      sens, run)
    X = sdb.fit_transform((ds_grad, None))
    ldb = LoadDataset(simulation.case_manager.dataset,
                      sens, run)
    X = ldb.fit_transform(X)
    assert not np.isnan(X[0].ica_components.values).any()
    assert X[0].ica_components.values.all()
    assert X[0].ica_component_properties.values.any()

    ds_saved = xr.load_dataset(simulation.case_manager.dataset)
    assert not np.isnan(ds_saved.ica_components.values).any()
    assert not ds_saved.ica_components.values.all()

    sens = 'mag'
    run = 0
    ds_mag, _ = select_sensors(ds, sens, run)
    assert ds_mag.channel_names.shape[0] == 102
    ds_mag['ica_components'].loc[:, :] = 5.
    ds_mag['ica_component_properties'].loc[
        dict(ica_component_property='mni_x')] = 6
    sdb = SaveDataset(simulation.case_manager.dataset,
                      sens, run)
    X = sdb.fit_transform((ds_mag, None))
    ldb = LoadDataset(simulation.case_manager.dataset,
                      sens, run)
    X = ldb.fit_transform(X)
    assert not np.isnan(X[0].ica_components.values).any()
    assert X[0].ica_components.values.all()
    assert X[0].ica_component_properties.values.any()

    ds_saved = xr.load_dataset(simulation.case_manager.dataset)
    assert not np.isnan(ds_saved.ica_components.values).any()
    assert ds_saved.ica_components.values.all()

    ica_comp = check_and_read_from_dataset(ds_saved, 'ica_components')
    assert (ds_saved.ica_components.values == ica_comp).all()

    check_and_write_to_dataset(
        ds_mag, 'ica_component_properties', np.array([9]*2),
        dict(ica_component_property="kurtosis"))
    assert (ds_mag.ica_component_properties.loc[
        dict(ica_component_property="kurtosis")].values == 9).all()
