import os.path as op
from pathlib import Path

import pytest
from megspikes.casemanager.casemanager import CaseManager
from megspikes.database.database import Database, LoadDataset, SaveDataset
from megspikes.detection.detection import (ComponentsSelection,
                                           DecompositionICA, PeakDetection)
from megspikes.localization.localization import ComponentsLocalization
from megspikes.simulation.simulation import Simulation
from megspikes.utils import PrepareData, ToTest
from sklearn.pipeline import Pipeline, FeatureUnion

sample_path = Path(op.dirname(__file__)).parent.parent
sample_path = sample_path / 'example'
sample_path.mkdir(exist_ok=True)

sim = Simulation(sample_path)
sim.load_mne_dataset()
sim.simulate_dataset(length=10)


@pytest.fixture(name="sensors")
def fixture_sensors():
    return ['grad', 'mag']


@pytest.fixture(name="runs")
def fixture_runs():
    return [0, 0]


@pytest.mark.pipeline
@pytest.mark.happy
@pytest.mark.slow
def test_pipeline(sensors, runs):
    case = CaseManager(root=sample_path, case='sample',
                       free_surfer=sim.subjects_dir)
    case.set_basic_folders()
    case.select_fif_file(case.run)
    case.prepare_forward_model()
    n_ica_components = 3
    db = Database(n_ica_components=n_ica_components)
    db.read_case_info(case.fif_file, case.fwd['ico5'])
    ds = db.make_empty_dataset()
    ds.to_netcdf(case.dataset)

    data_path = case.fif_file

    pipe_sensors = []
    pipe_runs = []

    for sens in ['grad', 'mag']:
        for run in range(4):
            pipe_runs.append(
                (f'run_{run}',
                 Pipeline([
                    ('prepare_data', PrepareData(sensors=sens, resample=200.)),
                    ('load_select_dataset',
                     LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                    ('select_ica_components', ComponentsSelection(run=run)),
                    ('detect_ica_peaks',
                     PeakDetection(prominence=2., width=1.)),
                    ('save_dataset',
                     SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                    ('test', ToTest())
                    ])))
        pipe_sensors.append(
            (sens,
             Pipeline([
                ('prepare_data',
                 PrepareData(data_file=data_path, sensors=sens)),
                ('select_dataset',
                 LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('ica_decomposition',
                 DecompositionICA(n_components=n_ica_components)),
                ('components_localization',
                 ComponentsLocalization(case=case, sensors=sens)),
                ('save_dataset',
                    SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('spikes_detection', FeatureUnion(pipe_runs))
                ])))
        pipe_runs = []
    pipe = Pipeline([
        ('make_clusters_library',  FeatureUnion(pipe_sensors))])
    _ = pipe.fit_transform(ds)
