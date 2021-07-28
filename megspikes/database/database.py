from typing import List, Union, Any, Tuple
from pathlib import Path
import xarray as xr
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import mne


class Database():
    def __init__(self, meg_data_length: int = 10_000,
                 n_fwd_sources: int = 20_000,
                 n_sensor_types: int = 2,
                 sensors: List[str] = ['grad', 'mag'],
                 n_sensors_by_types: List[int] = [204, 102],
                 n_ica_components: int = 20,
                 sfreq1: int = 1000,
                 sfreq2: int = 200,
                 n_runs: int = 4,
                 n_atoms: int = 3,
                 n_detected_peaks: int = 2000,
                 n_cleaned_peaks: int = 300,
                 atom_length: float = 0.5,  # seconds
                 n_clusters_library_timepoints: int = 2000,
                 n_times_cluster_epoch: int = 1000,
                 n_channels_grad: int = 204,
                 n_channels_mag: int = 102):
        self.meg_data_length = meg_data_length
        self.n_fwd_sources = n_fwd_sources
        self.n_sensor_types = n_sensor_types
        self.n_sensors_by_types = n_sensors_by_types
        self.sensors = sensors
        self.n_ica_components = n_ica_components
        self.sfreq1 = sfreq1  # Hz before downsampling
        self.sfreq2 = sfreq2  # Hz after downsampling
        self.n_atoms = n_atoms
        self.n_runs = n_runs
        self.n_detected_peaks = n_detected_peaks
        self.n_cleaned_peaks = n_cleaned_peaks
        self.atom_length = atom_length  # ms
        self.n_clusters_library_timepoints = n_clusters_library_timepoints
        self.n_times_cluster_epoch = n_times_cluster_epoch
        self.n_channels_grad = n_channels_grad
        self.n_channels_mag = n_channels_mag

    def make_empty_dataset(self) -> xr.Dataset:
        # --------- ICA decomposition --------- #
        n_samples_ica_sources = np.int32(
            self.meg_data_length / 1000 * self.sfreq1)
        ica_sources = xr.DataArray(
            np.zeros((self.n_sensor_types,
                      self.n_ica_components,
                      self.meg_data_length)),
            dims=("sensors", "ica_component", "time"),
            coords={
                "sensors": ['grad', 'mag'],
                "ica_component": np.arange(self.n_ica_components),
                "time": np.linspace(
                    0, self.meg_data_length, n_samples_ica_sources)
                },
            name="ica_sources",
            attrs=dict(
                description="",
                units="",
                )
            )

        ica_components = xr.DataArray(
            np.zeros((self.n_sensor_types, self.n_ica_components,
                      max(self.n_sensors_by_types))),
            dims=("sensors", "ica_component", "channels"),
            coords={
                "sensors": ['grad', 'mag'],
                "ica_component": np.arange(self.n_ica_components),
                },
            name="ica_components")

        ica_components_localization = xr.DataArray(
            np.zeros((self.n_sensor_types, self.n_ica_components, 3)),
            dims=("sensors", "ica_component", "mni_coordinates"),
            coords={
                "sensors": ['grad', 'mag'],
                "ica_component": np.arange(self.n_ica_components),
                "mni_coordinates": np.arange(3)
                },
            attrs={
                "n_grad": self.n_channels_grad,
                "n_mag": self.n_channels_mag},
            name="ica_components_localization")

        ica_components_gof = xr.DataArray(
            np.zeros((self.n_sensor_types, self.n_ica_components)),
            dims=("sensors", "ica_component"),
            coords={
                "sensors": ['grad', 'mag'],
                "ica_component": np.arange(self.n_ica_components)
                },
            name="ica_components_gof")

        ica_components_kurtosis = xr.DataArray(
            np.zeros((self.n_sensor_types, self.n_ica_components)),
            dims=("sensors", "ica_component"),
            coords={
                "sensors": ['grad', 'mag'],
                "ica_component": np.arange(self.n_ica_components)
                },
            name="ica_components_kurtosis")

        ica_components_selected = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_ica_components)),
            dims=("run", "sensors", "ica_component"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_component": np.arange(self.n_ica_components)
                },
            name="ica_components_selected")

        # --------- ICA peaks --------- #

        ica_peaks_timestamps = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "ica_timestamps"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_timestamps": np.arange(self.n_detected_peaks)
                },
            name="ica_peaks_timestamps")

        ica_peaks_localization = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks, 3)),
            dims=("run", "sensors", "ica_timestamps", "mni_coordinates"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_timestamps": np.arange(self.n_detected_peaks),
                "mni_coordinates": np.arange(3)
                },
            name="ica_peaks_localization")

        ica_peaks_subcorr = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "ica_timestamps"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_timestamps": np.arange(self.n_detected_peaks)
                },
            name="ica_peaks_subcorr")

        ica_peaks_selected = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "ica_timestamps"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_timestamps": np.arange(self.n_detected_peaks)
                },
            name="ica_peaks_selected")

        # --------- AlphaCSC decomposition --------- #

        u_hat = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_atoms, max(self.n_sensors_by_types))),
            dims=("run", "sensors", "atom", "channels"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "atom": np.arange(self.n_atoms),
                },
            attrs={
                "n_grad": self.n_channels_grad,
                "n_mag": self.n_channels_mag},
            name="u_hat")

        n_samples = np.int32(self.atom_length * self.sfreq2)
        v_hat = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_atoms, n_samples)),
            dims=("run", "sensors", "atom", "atom_times"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "atom": np.arange(self.n_atoms),
                "atom_times": np.linspace(
                    0, self.atom_length, n_samples)
                },
            name="v_hat")

        n_samples = np.int32(self.meg_data_length / 1000 * self.sfreq2)
        z_hat = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_atoms, n_samples)),
            dims=("run", "sensors", "atom", "z_hat_time"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "atom": np.arange(self.n_atoms),
                "z_hat_time": np.linspace(
                    0, self.meg_data_length, n_samples)
                },
            name="z_hat")

        alphacsc_components_localization = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types, self.n_atoms, 3)),
            dims=("run", "sensors", "atom", "mni_coordinates"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "atom": np.arange(self.n_atoms),
                "mni_coordinates": np.arange(3)
                },
            name="alphacsc_components_localization")

        alphacsc_components_gof = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types, self.n_atoms)),
            dims=("run", "sensors", "atom"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "atom": np.arange(self.n_atoms)
                },
            name="alphacsc_components_gof")

        # --------- AlphaCSC events alignment and clustering --------- #

        alphacsc_detections_timestamps = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "ica_timestamps"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_timestamps": np.arange(self.n_detected_peaks)
                },
            name="alphacsc_detections_timestamps")

        alphacsc_detections_goodness = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "ica_timestamps"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_timestamps": np.arange(self.n_detected_peaks)
                },
            name="alphacsc_detections_goodness")

        alphacsc_detections_atom = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "ica_timestamps"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_timestamps": np.arange(self.n_detected_peaks)
                },
            name="alphacsc_detections_atom")

        alphacsc_detections_z_values = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "ica_timestamps"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "ica_timestamps": np.arange(self.n_detected_peaks)
                },
            name="alphacsc_detections_z_values")

        # --------- Clusters library --------- #

        clusters_library_timestamps = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("clusters_library_detections"),
            coords={
                "clusters_library_detections": np.arange(
                    self.n_clusters_library_timepoints)
                },
            name="clusters_library_timestamps")

        clusters_library_atom = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("clusters_library_detections"),
            coords={
                "clusters_library_detections": np.arange(
                    self.n_clusters_library_timepoints)
                },
            name="clusters_library_atom")

        clusters_library_sensors = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("clusters_library_detections"),
            coords={
                "clusters_library_detections": np.arange(
                    self.n_clusters_library_timepoints)
                },
            name="clusters_library_sensors")

        clusters_library_run = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("clusters_library_detections"),
            coords={
                "clusters_library_detections": np.arange(
                    self.n_clusters_library_timepoints)
                },
            name="clusters_library_run")

        clusters_library_cluster_id = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("clusters_library_detections"),
            coords={
                "clusters_library_detections": np.arange(
                    self.n_clusters_library_timepoints)
                },
            name="clusters_library_cluster_id")

        # --------- Irritative zone prediction --------- #

        iz_predictions = xr.DataArray(
            np.zeros((4, self.n_fwd_sources)),
            dims=("prediction_type", "fwd_source"),
            coords={
                "prediction_type": ['auto_peak', 'auto_slope',
                                    'manual', 'resection']
                },
            name="iz_predictions")

        ds = xr.Dataset(data_vars={
            "ica_sources": ica_sources,
            "ica_components": ica_components,
            "ica_components_localization": ica_components_localization,
            "ica_components_gof": ica_components_gof,
            "ica_components_kurtosis": ica_components_kurtosis,
            "ica_components_selected": ica_components_selected,
            "ica_peaks_timestamps": ica_peaks_timestamps,
            "ica_peaks_localization": ica_peaks_localization,
            "ica_peaks_subcorr": ica_peaks_subcorr,
            "ica_peaks_selected": ica_peaks_selected,
            "alphacsc_u_hat": u_hat,
            "alphacsc_v_hat": v_hat,
            "alphacsc_z_hat": z_hat,
            "alphacsc_components_localization":
                alphacsc_components_localization,
            "alphacsc_components_gof": alphacsc_components_gof,
            "alphacsc_detections_timestamps": alphacsc_detections_timestamps,
            "alphacsc_detections_atom": alphacsc_detections_atom,
            "alphacsc_detections_z_values": alphacsc_detections_z_values,
            "alphacsc_detections_goodness": alphacsc_detections_goodness,
            "clusters_library_timestamps": clusters_library_timestamps,
            "clusters_library_atom": clusters_library_atom,
            "clusters_library_sensors": clusters_library_sensors,
            "clusters_library_run": clusters_library_run,
            "clusters_library_cluster_id": clusters_library_cluster_id,
            "iz_predictions": iz_predictions
            })
        return ds

    def read_case_info(self, fif_file_path: Union[str, Path],
                       fwd: mne.Forward) -> None:
        if not Path(fif_file_path).is_file():
            raise RuntimeError("Fif file was not found")
        fif_file = mne.io.read_raw_fif(fif_file_path, preload=False)
        info = fif_file.info
        self.n_channels_grad = len(mne.pick_types(info, meg='grad'))
        self.n_channels_mag = len(mne.pick_types(info, meg='mag'))
        self.meg_data_length = fif_file.n_times
        self.n_fwd_sources = sum([len(h['vertno']) for h in fwd['src']])
        del fif_file, fwd

    def select_sensors(self, ds: xr.Dataset, sensors: str,
                       run: int) -> xr.Dataset:
        return ds.sel(sensors=sensors, run=run).squeeze()


class LoadDataset(TransformerMixin, BaseEstimator):
    def __init__(self, dataset: Union[str, Path], sensors: str,
                 run: int) -> None:
        self.dataset = dataset
        if sensors == 'grad':
            self.sensors = 0
        else:
            self.sensors = 1
        self.run = run

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        selection = dict(run=self.run, sensors=self.sensors)
        ds = xr.load_dataset(self.dataset)
        return (ds[selection].squeeze(), X[1])


class SaveDataset(TransformerMixin, BaseEstimator):
    def __init__(self, dataset: Union[str, Path], sensors: str,
                 run: int) -> None:
        self.dataset = dataset
        if sensors == 'grad':
            self.sensors = 0
        else:
            self.sensors = 1
        self.run = run

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        selection = dict(run=self.run, sensors=self.sensors)
        ds = xr.load_dataset(self.dataset)
        # TODO: fix this
        for key, val in ds[selection].items():
            shape = ds[selection][key].shape
            if len(shape) == 1:
                ds[selection][key][:] = X[0][key]
            elif len(shape) == 2:
                ds[selection][key][:, :] = X[0][key]
            elif len(shape) == 3:
                ds[selection][key][:, :, :] = X[0][key]
        ds.to_netcdf(self.dataset, mode='a', format="NETCDF4",
                     engine="netcdf4")
        return X


class SaveFullDataset(TransformerMixin, BaseEstimator):
    def __init__(self, dataset: Union[str, Path]) -> None:
        self.dataset = dataset

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        X[0].to_netcdf(self.dataset, mode='a', format="NETCDF4",
                       engine="netcdf4")
        return X
