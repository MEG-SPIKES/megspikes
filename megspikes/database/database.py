import logging
import warnings
from pathlib import Path
import traceback
from typing import Any, Dict, List, Tuple, Union

import mne
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin


class Database():
    def __init__(self,
                 sensors: List[str],
                 channel_names: List[str],
                 channels_by_sensors: Dict[str, List[int]],
                 fwd_sources: List[List[int]]):
        self.fwd_sources = fwd_sources
        self.sensors = sensors
        self.channels_by_sensors = channels_by_sensors
        self.channel_names = channel_names

    def make_aspire_alphacsc_dataset(self,
                                     times: np.ndarray,
                                     n_ica_components: int,
                                     n_atoms: int,
                                     atom_length: float,
                                     n_runs: int,
                                     sfreq: float = 200.,
                                     ) -> xr.Dataset:
        """Make empty xr.Dataset to run ASPIRE AlphaCSC pipeline.

        Parameters
        ----------
        times : np.ndarray
            mne.io.Raw times in seconds
        n_ica_components : int
            number of ICA components that will be exprected for each sensors
            type
        n_atoms : int
            number of AlphaCSC atoms that will be extracted in the each run
        atom_length : float
            AlphaCSC atom's length in seconds
        n_runs : int
            number of the pipeline repetitions
        sfreq : float, optional
            sample frequency, by default 200.

        Returns
        -------
        xr.Dataset
            empty dataset prepared for the pipeline
        """
        # ---- Prepare dimensions and coordinates ---- #
        # time coordinate
        t = xr.DataArray(
            data=times,
            dims=("time"),
            attrs={
                'sfreq': sfreq,
                # 'units': 'seconds'
            },
            name='time')
        meg_data_length = len(t)

        # runs
        runs = np.arange(n_runs)

        # channels and sensors
        sensors = self.sensors
        channel_names_list = self.channel_names
        channels = np.arange(len(channel_names_list))
        grad_inx = self.channels_by_sensors['grad']
        mag_inx = self.channels_by_sensors['mag']

        # detected events properties
        detection_property_coord = xr.DataArray(
            data=['ica_detection', 'ica_component', 'mni_x', 'mni_y', 'mni_z',
                  'subcorr', 'selected_for_alphacsc', 'alphacsc_detection',
                  'alphacsc_atom', 'ica_alphacsc_aligned'],
            dims=('detection_property'),
            attrs={
                'sfreq': sfreq,
                'ica_component_description': (
                    'Index of the ICA component where the '
                    'timestamp was detected'),
                'mni_x_units': 'mm',
                'mni_y_units': 'mm',
                'mni_z_units': 'mm',
                'subcorr': ('The measurement of the RAP MUSIC quality. '
                            'Larger values correspond to the better solution'),
                'subcorr_units': 'arbitrary units',
                'selected_for_alphacsc': 'ICA peaks selected for AlphaCSC',
                'alphacsc_detection': 'Final AlphaCSC detections',
                'alphacsc_atom': ('AlphaCSC atom that corresponds to the '
                                  'AlphaCSC detection'),
                'ica_alphacsc_aligned': ('Alignment of the ICA components '
                                         'that was done using AlphaCSC')
            },
            name='detection_property_coord')

        atoms_library_properties = xr.DataArray(
            data=['library_detection', 'library_cluster',
                  'library_run', 'library_sensors'],
            dims=('atoms_library_property'),
            name='atoms_library_properties')

        # ica components
        ica_components_coords = np.arange(n_ica_components)
        ica_component_properties_coords = xr.DataArray(
            data=['mni_x', 'mni_y', 'mni_z', 'gof', 'kurtosis'],
            dims=('ica_component_property'),
            attrs={
                'mni_x_units': 'mm',
                'mni_y_units': 'mm',
                'mni_z_units': 'mm',
                'gof': ('The measurement of the dipole fitting quality.'
                        'Larger values correspond to the better solution'),
                'gof_units': 'percentage between [0, 100]'
            },
            name='ica_component_properties_coords')

        # alphacsc atoms
        n_alphacsc_atoms = n_atoms
        alphacsc_atoms_coords = np.arange(n_alphacsc_atoms)
        atom_length = int(atom_length * sfreq)
        atom_v_times = np.linspace(0, atom_length, atom_length)
        alphacsc_atoms_properties_coords = xr.DataArray(
            data=['mni_x', 'mni_y', 'mni_z', 'gof', 'goodness', 'selected'],
            dims=('alphacsc_atom_property'),
            attrs={
                'mni_x_units': 'mm',
                'mni_y_units': 'mm',
                'mni_z_units': 'mm',
                'gof': ('The measurement of the dipole fitting quality.'
                        'Larger values correspond to the better solution'),
                'gof_units': 'percentage between [0, 100]',
                'goodness': 'comprehensive assessment atoms quality',
                'selected': 'Atoms selected for clusters library'
            },
            name='alphacsc_atoms_properties_coords')

        # ---- Prepare dataarrays ---- #

        channel_names = xr.DataArray(
            data=channel_names_list,
            dims=("channel"),
            coords={"channel": channels},
            attrs={
                'grad': grad_inx,
                'mag': mag_inx,
            },
            name="channel_names")

        detection_properties = xr.DataArray(
            data=np.zeros(
                (n_runs, len(sensors),
                 len(detection_property_coord), meg_data_length)),
            dims=("run", "sensors", "detection_property", "time"),
            coords={
                "run": runs,
                "sensors": sensors,
                "detection_property": detection_property_coord,
                "time": t
            },
            name="detection_properties")

        ica_sources = xr.DataArray(
            data=np.zeros((len(sensors), n_ica_components, meg_data_length)),
            dims=("sensors", "ica_component", "time"),
            coords={
                "sensors": sensors,
                "ica_component": ica_components_coords,
                "time": t
            },
            attrs={
                'units': 'AU'
            },
            name="ica_sources")

        ica_components = xr.DataArray(
            data=np.zeros((n_ica_components, len(channels))),
            dims=("ica_component", "channel"),
            coords={
                "ica_component": ica_components_coords,
                "channel": channels
            },
            attrs={
                'grad': grad_inx,
                'mag': mag_inx,
            },
            name="ica_components")

        ica_component_properties = xr.DataArray(
            data=np.zeros(
                (len(sensors), n_ica_components,
                 len(ica_component_properties_coords))),
            dims=("sensors", "ica_component", "ica_component_property"),
            coords={
                "sensors": sensors,
                "ica_component": ica_components_coords,
                "ica_component_property": ica_component_properties_coords,
            },
            name="ica_component_properties")

        ica_component_selection = xr.DataArray(
            data=np.zeros((n_runs, len(sensors), n_ica_components)),
            dims=("run", "sensors", "ica_component"),
            coords={
                "run": runs,
                "sensors": sensors,
                "ica_component": ica_components_coords,
            },
            name="ica_component_selection")

        alphacsc_z_hat = xr.DataArray(
            data=np.zeros((
                n_runs, len(sensors), n_alphacsc_atoms,
                meg_data_length)),
            dims=("run", "sensors", "alphacsc_atom", "time"),
            coords={
                "run": runs,
                "sensors": sensors,
                "alphacsc_atom": alphacsc_atoms_coords,
                "time": t
            },
            attrs={
                'units': 'AU'
            },
            name="alphacsc_z_hat")

        alphacsc_v_hat = xr.DataArray(
            data=np.zeros(
                (n_runs, len(sensors), n_alphacsc_atoms, atom_length)),
            dims=("run", "sensors", "alphacsc_atom", "atom_v_time"),
            coords={
                "run": runs,
                "sensors": sensors,
                "alphacsc_atom": alphacsc_atoms_coords,
                "atom_v_time": atom_v_times
            },
            attrs={
                'units': 'AU'
            },
            name="alphacsc_v_hat")

        alphacsc_u_hat = xr.DataArray(
            data=np.zeros((n_runs, n_alphacsc_atoms, len(channels))),
            dims=("run", "alphacsc_atom", "channel"),
            coords={
                "run": runs,
                "alphacsc_atom": alphacsc_atoms_coords,
                "channel": channels
            },
            attrs={
                'grad': grad_inx,
                'mag': mag_inx,
                'units': 'AU'
            },
            name="alphacsc_u_hat")

        alphacsc_atoms_properties = xr.DataArray(
            data=np.zeros(
                (n_runs, len(sensors), n_alphacsc_atoms,
                 len(alphacsc_atoms_properties_coords))),
            dims=("run", "sensors", "alphacsc_atom",
                  "alphacsc_atom_property"),
            coords={
                "run": runs,
                "sensors": sensors,
                "alphacsc_atom": alphacsc_atoms_coords,
                "alphacsc_atom_property": alphacsc_atoms_properties_coords,
            },
            name="alphacsc_atoms_properties")

        alphacsc_atoms_library_properties = xr.DataArray(
            data=np.zeros((len(atoms_library_properties), meg_data_length)),
            dims=("atoms_library_property", "time"),
            coords={
                "atoms_library_property": atoms_library_properties,
                "time": t
            },
            name="alphacsc_atoms_library_properties")

        # ---- Create dataset ---- #

        ds = xr.merge([
            channel_names,
            ica_sources,
            ica_components,
            ica_component_properties,
            ica_component_selection,
            detection_properties,
            alphacsc_z_hat,
            alphacsc_v_hat,
            alphacsc_u_hat,
            alphacsc_atoms_properties,
            alphacsc_atoms_library_properties
        ])
        return ds

    def make_clusters_dataset(self, times: np.ndarray, n_clusters: int,
                              evoked_length: float, sfreq: float = 1000.
                              ) -> xr.Dataset:
        """Make empty xr.Dataset to run clusters' localization and irritative
           area prediction.

        Parameters
        ----------
        times : np.ndarray
            mne.io.Raw times in seconds
        n_clusters : int
            number of the detected clusters
        evoked_length : float
            length of the cluster averages in seconds
        sfreq : float, optional
            sample frequency, by default 1000.

        Returns
        -------
        xr.Dataset
            empty dataset prepared for the pipeline
        """
        # ---- Prepare dimensions and coordinates ---- #
        cluster = xr.DataArray(
            data=np.arange(n_clusters),
            dims=("cluster"),
            name='cluster')

        time = xr.DataArray(
            data=times,
            dims=("time"),
            attrs={
                'sfreq': sfreq,
                # 'units': 'seconds'
            },
            name='time')

        time_evoked = xr.DataArray(
            data=np.linspace(0, evoked_length, int(evoked_length * sfreq)),
            dims=("time_evoked"),
            attrs={
                'sfreq': sfreq,
                # 'units': 'seconds'
            },
            name='time_evoked')

        n_fwd_sources = sum([len(h) for h in self.fwd_sources])
        source = xr.DataArray(
            data=np.arange(n_fwd_sources),
            dims=("source"),
            attrs={
                'lh_vertno': self.fwd_sources[0],
                'rh_vertno': self.fwd_sources[1],
            },
            name='source')

        # channels
        sensors = self.sensors
        channel_names_list = self.channel_names
        channels = np.arange(len(channel_names_list))
        grad_inx = self.channels_by_sensors['grad']
        mag_inx = self.channels_by_sensors['mag']

        detection_property_coords = xr.DataArray(
            data=['detection', 'cluster', 'sensor', 'run'],
            dims=('detection_property'),
            attrs={},
            name='detection_property')

        cluster_property_coords = xr.DataArray(
            data=['cluster_id', 'sensors', 'run', 'atom', 'pipeline_type',
                  'n_events', 'time_baseline', 'time_slope', 'time_peak',
                  'selected_for_iz_prediction'],
            dims=('cluster_property'),
            attrs={
                'grad_index': 0,
                'mag_index': 1},
            name='cluster_property')

        iz_prediction_timepoints_coords = xr.DataArray(
            data=['baseline', 'slope', 'peak'],
            dims=('iz_prediction_timepoint'),
            name='iz_prediction_timepoint')

        # ---- Prepare dataarrays ---- #

        channel_names = xr.DataArray(
            data=channel_names_list,
            dims=("channel"),
            coords={"channel": channels},
            attrs={
                'grad': grad_inx,
                'mag': mag_inx,
            },
            name="channel_names")

        spike = xr.DataArray(
            data=np.zeros((len(times), len(detection_property_coords))),
            dims=("time", "detection_property"),
            coords={
                "time": time,
                "detection_property": detection_property_coords
            },
            attrs={
                'sfreq': sfreq,
            },
            name="spike")

        cluster_properties = xr.DataArray(
            data=np.zeros((n_clusters, len(cluster_property_coords))),
            dims=("cluster", "cluster_property"),
            coords={
                "cluster": cluster,
                "cluster_property": cluster_property_coords
            },
            name="cluster_properties")

        mne_localization = xr.DataArray(
            data=np.zeros(
                (len(sensors), n_clusters, len(source), len(time_evoked))),
            dims=("sensors", "cluster", "source", "time_evoked"),
            coords={
                "sensors": sensors,
                "cluster": cluster,
                "source": source,
                "time_evoked": time_evoked
            },
            name="mne_localization")

        evoked = xr.DataArray(
            data=np.zeros((n_clusters, len(channels), len(time_evoked))),
            dims=("cluster", "channel", "time_evoked"),
            coords={
                "cluster": cluster,
                "channel": channels,
                "time_evoked": time_evoked
            },
            name="evoked")

        iz_prediction = xr.DataArray(
            data=np.zeros(
                (len(source), 3)),
            dims=("source", "iz_prediction_timepoint"),
            coords={
                "source": source,
                "iz_prediction_timepoint": iz_prediction_timepoints_coords
            },
            name="iz_prediction")

        fwd_mni_coordinates = xr.DataArray(
            data=np.zeros(
                (len(source), 3)),
            dims=("source", "mni_coordinate"),
            coords={
                "source": source,
                "mni_coordinate": ['x', 'y', 'z']
            },
            attrs={
                "mni_coordinate_units": "mm"
            },
            name="fwd_mni_coordinates")

        # ---- Create dataset ---- #

        ds = xr.merge([
            channel_names,
            spike,
            cluster_properties,
            mne_localization,
            evoked,
            iz_prediction,
            fwd_mni_coordinates
        ])
        return ds

    def select_sensors(self, ds: xr.Dataset, sensors: str,
                       run: int) -> xr.Dataset:
        ds_subset, _ = select_sensors(ds, sensors, run)
        return ds_subset


class PrepareClustersDataset(BaseEstimator, TransformerMixin):
    """Prepare xr.Dataset for the cluster's localization procedure."""

    def __init__(self, fif_file, fwd,
                 detection_sfreq: float = 200.,
                 evoked_length: float = 1.,
                 ) -> None:
        self.fif_file = fif_file
        self.fwd = fwd  # ico5
        self.evoked_length = evoked_length
        self.detection_sfreq = detection_sfreq

    def fit(self, X: Tuple[dict, mne.io.Raw], y=None):
        self.sfreq = X[1].info['sfreq']

        self.spikes = X[0]['spikes']
        for i in ['spike_clusters', 'spike_sensors', 'spike_runs']:
            if i in X[0].keys():
                setattr(self, i, X[0][i])
            else:
                setattr(self, i, None)

        if self.spike_clusters is None:
            logging.info("PrepareClustersDataset: No clusters provided")
            self.spike_clusters = np.zeros_like(self.spikes, dtype=np.int32)
        if self.spike_sensors is None:
            logging.info("PrepareClustersDataset: No sensors provided")
            self.spike_sensors = np.zeros_like(self.spikes, dtype=np.int32)
        if self.spike_runs is None:
            logging.info("PrepareClustersDataset: No spike runs provided")
            self.spike_runs = np.zeros_like(self.spikes, dtype=np.int32)
        return self

    def transform(self, X) -> Tuple[xr.Dataset, mne.io.Raw]:
        selected_clusters = np.unique(self.spike_clusters)
        database = read_meg_info_for_database(self.fif_file, self.fwd)
        ds = database.make_clusters_dataset(
            X[1].times, len(selected_clusters), self.evoked_length,
            self.sfreq)

        # resample timestamps
        resampled_spikes = self.resample_timestamps(
            self.spikes, self.detection_sfreq, self.sfreq)

        # write detections
        spikes = np.zeros_like(ds.time.values)
        spikes[resampled_spikes] = 1
        check_and_write_to_dataset(
            ds, 'spike', spikes, dict(detection_property='detection'))
        spike_clusters = np.zeros_like(ds.time.values)
        spike_clusters[resampled_spikes] = self.spike_clusters
        check_and_write_to_dataset(
            ds, 'spike', spike_clusters, dict(detection_property='cluster'))
        spike_sensors = np.zeros_like(ds.time.values)
        spike_sensors[resampled_spikes] = self.spike_sensors
        check_and_write_to_dataset(
            ds, 'spike', spike_sensors, dict(detection_property='sensor'))
        spike_runs = np.zeros_like(ds.time.values)
        spike_runs[resampled_spikes] = self.spike_runs
        check_and_write_to_dataset(
            ds, 'spike', spike_runs, dict(detection_property='run'))

        # write cluster properties
        runs = np.zeros_like(selected_clusters)
        sensors = np.zeros_like(selected_clusters)
        n_events = np.zeros_like(selected_clusters)
        for i, cl in enumerate(selected_clusters):
            cluster_mask = self.spike_clusters == cl
            assert len(np.unique(self.spike_runs[cluster_mask])) == 1
            assert len(np.unique(self.spike_sensors[cluster_mask])) == 1
            assert len(np.unique(self.spike_runs[cluster_mask])) == 1
            runs[i] = self.spike_runs[cluster_mask][0]
            sensors[i] = self.spike_sensors[cluster_mask][0]
            n_events[i] = len(np.where(cluster_mask)[0])

        check_and_write_to_dataset(
            ds, 'cluster_properties', selected_clusters,
            dict(cluster_property='cluster_id'))
        check_and_write_to_dataset(
            ds, 'cluster_properties', runs,
            dict(cluster_property='run'))
        check_and_write_to_dataset(
            ds, 'cluster_properties', sensors,
            dict(cluster_property='sensors'))
        check_and_write_to_dataset(
            ds, 'cluster_properties', n_events,
            dict(cluster_property='n_events'))
        return (ds, X[1])

    def resample_timestamps(self, timestamps: np.ndarray,
                            from_sfreq: float = 200.,
                            to_sfreq: float = 1000.) -> np.ndarray:
        return np.int32((timestamps / from_sfreq) * to_sfreq)


class PrepareAspireAlphacscDataset(BaseEstimator, TransformerMixin):
    """Prepare xr.Dataset for the ASPIRE AlphaCSC detection pipeline."""

    def __init__(self,
                 fif_file: Union[str, Path],
                 fwd: mne.Forward,
                 atoms_width: float = 1.,
                 n_runs: int = 4,
                 n_ica_components: int = 20,
                 n_atoms: int = 3) -> None:
        self.fif_file = fif_file
        self.fwd = fwd  # ico5
        self.atoms_width = atoms_width
        self.n_runs = n_runs
        self.n_ica_components = n_ica_components
        self.n_atoms = n_atoms

    def fit(self, X: mne.io.Raw, y=None):
        return self

    def transform(self, X: mne.io.Raw) -> Tuple[xr.Dataset, mne.io.Raw]:
        database = read_meg_info_for_database(self.fif_file, self.fwd)
        ds = database.make_aspire_alphacsc_dataset(
            times=X.times,
            n_ica_components=self.n_ica_components,
            n_atoms=self.n_atoms,
            atom_length=self.atoms_width,
            n_runs=self.n_runs,
            sfreq=X.info['sfreq'])
        return (ds, X)


class LoadDataset(TransformerMixin, BaseEstimator):
    def __init__(self, dataset: Union[str, Path], sensors: Union[str, None],
                 run: Union[int, None]) -> None:
        self.dataset = dataset
        self.sensors = sensors
        self.run = run

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        full_ds = xr.load_dataset(self.dataset)
        if (self.sensors is not None) & (self.run is not None):
            ds, _ = select_sensors(full_ds, self.sensors, self.run)
        else:
            ds = full_ds
        if isinstance(X, Tuple):
            return (ds, X[1])
        elif isinstance(X, mne.io.Raw):
            return (ds, X)
        else:
            raise RuntimeError("LoadDataset: Wrong input")


class SaveDataset(TransformerMixin, BaseEstimator):
    """Save merge subset of the database in the full dataset
    """
    def __init__(self, dataset: Union[str, Path],
                 sensors: Union[str, None] = None,
                 run: Union[int, None] = None,
                 rewrite_previous_results=False) -> None:
        self.dataset = dataset
        self.sensors = sensors
        self.run = run
        self.rewrite_previous_results = rewrite_previous_results

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        if isinstance(self.sensors, str) & isinstance(self.run, int):
            ds = xr.load_dataset(self.dataset)
            self.update_selected_fields(X[0], ds)
            ds.to_netcdf(self.dataset, mode='a', format="NETCDF4",
                         engine="netcdf4")
        elif isinstance(self.sensors, str) | isinstance(self.run, int):
            raise RuntimeError(f'Sensors {type(self.sensors)} or run '
                               f'{type(self.run)} have a wrong type.')
        else:
            if self.dataset.is_file() & (not self.rewrite_previous_results):
                raise RuntimeError(
                    'Results dataset exists and you try to overwrite it')
            else:
                if self.rewrite_previous_results | (not self.dataset.is_file()):
                    X[0].to_netcdf(self.dataset, mode='w', format="NETCDF4",
                                   engine="netcdf4")
                else:
                    try:
                        X[0].to_netcdf(self.dataset, mode='a', format="NETCDF4",
                                       engine="netcdf4")
                    except:
                        raise RuntimeError(
                            'Results dataset exists and you try to add'
                            'inconsistent data in it.')
        return X

    def update_selected_fields(self, from_ds: xr.Dataset, to_ds: xr.Dataset):
        # TODO: avoid manual update
        # Won't working because selection is no applicable to all ds variables
        # ds.loc[selection] = ds_grad
        selection_sens = dict(sensors=self.sensors)
        to_ds['ica_sources'].loc[selection_sens] = from_ds.ica_sources
        to_ds['ica_component_properties'].loc[
            selection_sens] = from_ds.ica_component_properties

        selection_ch = to_ds.attrs[self.sensors]
        to_ds['ica_components'].loc[:, selection_ch] = from_ds.ica_components
        selection_run_sens = dict(
            run=self.run, sensors=self.sensors)
        to_ds['ica_component_selection'].loc[
            selection_run_sens] = from_ds.ica_component_selection
        to_ds['detection_properties'].loc[
            selection_run_sens] = from_ds.detection_properties
        to_ds['alphacsc_z_hat'].loc[
            selection_run_sens] = from_ds.alphacsc_z_hat
        to_ds['alphacsc_v_hat'].loc[
            selection_run_sens] = from_ds.alphacsc_v_hat
        to_ds['alphacsc_atoms_properties'].loc[
            selection_run_sens] = from_ds.alphacsc_atoms_properties

        selection_run_ch = dict(
            run=self.run, channel=selection_ch)
        to_ds['alphacsc_u_hat'].loc[
            selection_run_ch] = from_ds.alphacsc_u_hat


class ReadDetectionResults(TransformerMixin, BaseEstimator):
    """Read detection pipeline results to strart cluster pipeline."""

    def __init__(self) -> None:
        pass

    def fit(self, X: Tuple[xr.Dataset, mne.io.Raw], y=None):
        return self

    def transform(self, X: Tuple[xr.Dataset, mne.io.Raw]
                  ) -> Tuple[dict, mne.io.Raw]:
        spikes = check_and_read_from_dataset(
            X[0], 'alphacsc_atoms_library_properties',
            dict(atoms_library_property='library_detection'))
        clusters = check_and_read_from_dataset(
            X[0], 'alphacsc_atoms_library_properties',
            dict(atoms_library_property='library_cluster'))
        sensors = check_and_read_from_dataset(
            X[0], 'alphacsc_atoms_library_properties',
            dict(atoms_library_property='library_sensors'))
        runs = check_and_read_from_dataset(
            X[0], 'alphacsc_atoms_library_properties',
            dict(atoms_library_property='library_run'))
        mask = spikes != 0
        detections = {
            'spikes': np.where(mask != 0)[0],
            'spike_clusters': clusters[mask],
            'spike_sensors': sensors[mask],
            'spike_runs': runs[mask],
        }
        return (detections, X[1])


def select_sensors(ds: xr.Dataset, sensors: str,
                   run: int) -> xr.Dataset:
    channels = ds.channel_names.attrs[sensors]
    selection = dict(run=run, sensors=sensors, channel=channels)
    return ds.loc[selection], selection


def check_and_read_from_dataset(ds: xr.Dataset, da_name: str,
                                selection: Union[Dict[Any, str], None] = None,
                                dtype: type = np.float64) -> np.ndarray:
    if not isinstance(da_name, str):
        raise RuntimeError(f"{da_name} has type {type(da_name)}")
    assert da_name in ds.data_vars, (
        f"{da_name} not in dataset")

    if isinstance(selection, dict):
        data = ds[da_name].loc[selection].values.copy()
        try:
            if np.max(data) == np.min(data):
                warnings.warn(
                    f"{da_name}.loc[{selection}] values are all the same")
        except Exception:
            warnings.warn(
                f"{da_name}.loc[{selection}] is invalid")
            print(traceback.format_exc())
    else:
        data = ds[da_name].values.copy()
        if np.max(data) == np.min(data):
            warnings.warn(
                f"{da_name} values are all the same")
        else:
            assert data.any(), (
                f"{da_name} values are all zeros")
    return dtype(data)


def check_and_write_to_dataset(ds: xr.Dataset, da_name: str,
                               variable: np.ndarray,
                               selection: Union[Dict[Any, Union[
                                   str, List[str]]], None] = None
                               ) -> None:
    # NOTE: there is no output, operation is done inplace
    if not isinstance(da_name, str):
        raise RuntimeError(f"{da_name} has type {type(da_name)}")
    assert da_name in ds.data_vars, (
        f"{da_name} not in dataset")

    if isinstance(selection, dict):
        assert ds[da_name].loc[selection].shape == variable.shape, (
            f"Wrong shape of the variable to write in {da_name}")
        ds[da_name].loc[selection] = variable
    else:
        assert ds[da_name].shape == variable.shape, (
            f"Wrong shape of the variable to write in {da_name}")
        ds[da_name].values = variable


def read_meg_info_for_database(fif_file_path: Union[str, Path],
                               fwd: mne.Forward) -> Database:
    if not Path(fif_file_path).is_file():
        raise RuntimeError("Fif file was not found")
    info = mne.io.read_info(fif_file_path)
    # TODO: other sensors types
    sensors = ['grad', 'mag']
    channels_by_sensors = {}
    for sens in sensors:
        channels_by_sensors[sens] = mne.pick_types(info, meg=sens)
    channel_names = info['ch_names']
    # first: left hemi, second: right hemi
    fwd_sources = [h['vertno'] for h in fwd['src']]
    del fwd
    return Database(sensors, channel_names, channels_by_sensors, fwd_sources)
