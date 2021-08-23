import os.path as op
from pathlib import Path

import numpy as np
import pytest
from megspikes.simulation.simulation import Simulation
import mne

mne.set_log_level("ERROR")


@pytest.fixture(name="sample_path")
def sample_path():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_simulation'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.mark.happy
def test_simulation(sample_path):
    sim = Simulation(sample_path)
    sim.simulate_dataset()
    assert len(sim.clusters) == len(sim.detections)
    raw_fif = mne.io.read_raw_fif(sim.case_manager.fif_file)
    assert (raw_fif.times == sim.raw_simulation.times).all()


@pytest.mark.parametrize(
    "n_events,simultaneous,events",
    [([5, 0, 0, 0], [False]*4, np.vstack(
        [np.arange(5)*1000,
         np.zeros(5),
         np.repeat(1, 5)]).T),
     ([5, 5, 0, 0], [False]*4, np.vstack(
        [np.arange(10)*1000,
         np.zeros(10),
         np.array([1]*5 + [2]*5)]).T),
     ([5, 5, 5, 5], [False]*4, np.vstack(
        [np.arange(20)*1000,
         np.zeros(20),
         np.array([1]*5 + [2]*5 + [3]*5 + [4]*5)]).T),
     ([5, 5, 0, 0], [True, False, False, False], np.vstack(
        [np.array([i*1000 for i in range(5)]*2).flatten(),
         np.zeros(10),
         np.array([1]*5 + [2]*5)]).T),
     ])
def test_events_simulation(n_events, simultaneous, events):
    sim = Simulation(sample_path)
    ev = sim._prepare_events(n_events, simultaneous)
    assert (ev == events).all()


@pytest.mark.parametrize(
    "events",
    [(np.vstack(
        [np.arange(5)*1000,
         np.zeros(5),
         np.repeat(1, 5)]).T),
     (np.vstack(
        [np.array([i*1000 for i in range(5)]*2).flatten(),
         np.zeros(10),
         np.array([1]*5 + [2]*5)]).T),
     ])
def test_source_simulation(events):
    sim = Simulation(sample_path)
    _, fwd, _ = sim._read_mne_sample_dataset()
    source_simulator = sim._simulate_sources(events, fwd['src'], 1/1000)
    assert (source_simulator._events == events).all()
