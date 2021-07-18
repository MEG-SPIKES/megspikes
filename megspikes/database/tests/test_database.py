# -*- coding: utf-8 -*-
from megspikes.database.database import Database


def test_database():
    db = Database()
    ds = db.make_empty_dataset()
    ds_grad = ds.sel(
        sensors='grad', decomposition_sensors_type='grad', run=0).squeeze()
    assert ds_grad['ica_components'].values.shape == (20, 204)
