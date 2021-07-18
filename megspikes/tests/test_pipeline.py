import os.path as op
from pathlib import Path

import pytest
from megspikes.casemanager.casemanager import CaseManager
from megspikes.database.database import Database
from megspikes.detection.detection import (ComponentsSelection,
                                           DecompositionICA, PeakDetection)
from megspikes.localization.localization import ComponentsLocalization
from megspikes.simulation.simulation import Simulation
from megspikes.utils import PrepareData
from sklearn.pipeline import make_pipeline

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
    db = Database(n_ica_components=2)
    db.read_case_info(case.fif_file, case.fwd['ico5'])
    ds = db.make_empty_dataset()

    for sens, run in zip(sensors, runs):
        ds_sens = db.select_sensors(ds, sens, run)
        case.prepare_forward_model(sensors=sens)
        pipe = make_pipeline(
            PrepareData(case.fif_file, sens),
            DecompositionICA(n_components=2),
            ComponentsLocalization(case),
            ComponentsSelection(),
            PeakDetection(prominence=2., width=1.))
        _ = pipe.fit_transform(ds_sens)
