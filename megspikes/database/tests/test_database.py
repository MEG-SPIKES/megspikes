import numpy as np
from megspikes.database.database import Database


def test_database():
    db = Database()
    ds = db.make_empty_dataset()
    ds_grad = db.select_sensors(ds, 'grad', 0)
    _ = db.select_sensors(ds, 'mag', 0)
    ds_grad['ica_components'][:, :] = np.ones((20, 204))
    assert ds['ica_components'].any()
