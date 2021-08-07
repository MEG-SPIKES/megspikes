import os.path as op
from pathlib import Path

import pytest
from megspikes.database.database import Database
from megspikes.pipeline import aspike_alphacsc_pipeline


@pytest.fixture(scope="module", name='test_sample_path')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent
    sample_path = sample_path / 'tests_data' / 'test_pipeline'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.mark.pipeline
@pytest.mark.happy
@pytest.mark.slow
def test_pipeline(simulation):
    n_ica_components = 3
    n_ica_peaks = 50
    resample = 200.
    n_atoms = 2  # FIXME: one atom cause bugs
    z_hat_threshold = 1.
    z_hat_threshold_min = 0.1

    db = Database(n_ica_components=n_ica_components,
                  n_atoms=n_atoms)
    case = simulation.case_manager
    db.read_case_info(case.fif_file, case.fwd['ico5'], resample)
    ds = db.make_empty_dataset()
    ds.to_netcdf(case.dataset)  # save empty dataset

    pipe = aspike_alphacsc_pipeline(
        case, n_ica_components=n_ica_components, resample=resample,
        n_ica_peaks=n_ica_peaks, n_atoms=n_atoms,
        z_hat_threshold=z_hat_threshold,
        z_hat_threshold_min=z_hat_threshold_min,
        pipe_names=db.pipelines_names)
    _ = pipe.fit_transform(None)
