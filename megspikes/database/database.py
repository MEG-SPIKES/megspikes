# from typing import Callable, Iterator, Union, Optional, List
from typing import List
import xarray as xr
import numpy as np


class Database():
    def __init__(self, meg_data_length: int = 10_000,
                 n_fwd_sources: int = 20_000,
                 n_sensor_types: int = 2, sensors: List[str] = ['grad', 'mag'],
                 n_sensors_by_types: List[int] = [204, 102],
                 n_ica_components: int = 20, sfreq1: int = 1000.,
                 sfreq2: int = 200, n_runs: int = 4, n_atoms: int = 3,
                 n_detected_peaks: int = 0, n_cleaned_peaks: int = 0,
                 atom_length: int = 1000,
                 n_clusters_library: int = 1,
                 n_clusters_library_timepoints: int = 0,
                 n_times_cluster_epoch: int = 1000):
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
        self.n_clusters_library = n_clusters_library
        self.n_clusters_library_timepoints = n_clusters_library_timepoints
        self.n_times_cluster_epoch = n_times_cluster_epoch

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
            name="ica_sources")

        ica_comp_vars = {}
        for sens, n_channels in zip(self.sensors,
                                    self.n_sensors_by_types):
            ica_comp_vars[sens] = xr.DataArray(
                np.zeros((self.n_ica_components, n_channels)),
                dims=("ica_component", f"channels_{sens}"),
                coords={
                    "ica_component": np.arange(self.n_ica_components),
                    f"channels_{sens}": np.arange(n_channels)
                    },
                name=f"ica_components_{sens}")

        ica_components = xr.Dataset(data_vars=ica_comp_vars)

        ica_components = ica_components.to_stacked_array(
            "channels", sample_dims=["ica_component"],
            variable_dim="decomposition_sensors_type")

        ica_components_localization = xr.DataArray(
            np.zeros((self.n_sensor_types, self.n_ica_components, 3)),
            dims=("sensors", "ica_component", "mni_coordinates"),
            coords={
                "sensors": ['grad', 'mag'],
                "ica_component": np.arange(self.n_ica_components),
                "mni_coordinates": np.arange(3)
                },
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
                },
            name="ica_peaks_timestamps")

        ica_peaks_localization = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types, 3)),
            dims=("run", "sensors", "mni_coordinates"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "mni_coordinates": np.arange(3)
                },
            name="ica_peaks_localization")

        ica_peaks_subcorr = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "subcorr"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                },
            name="ica_peaks_subcorr")

        ica_peaks_selected = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "selected_peaks"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                },
            name="ica_peaks_selected")

        # --------- AlphaCSC decomposition --------- #

        u_hat_vars = {}
        for sens, n_channels in zip(self.sensors,
                                    self.n_sensors_by_types):
            u_hat_vars[sens] = xr.DataArray(
                np.zeros((self.n_runs, self.n_atoms, n_channels)),
                dims=("run", "atom", f"channels_{sens}"),
                coords={
                    "run": np.arange(self.n_runs),
                    "atom": np.arange(self.n_atoms),
                    f"channels_{sens}": np.arange(n_channels)
                    },
                name=f"u_hat_{sens}")

        u_hat = xr.Dataset(data_vars=u_hat_vars)

        u_hat = u_hat.to_stacked_array(
            "channels", sample_dims=["run", "atom"],
            variable_dim="decomposition_sensors_type")

        n_samples = np.int32(self.atom_length / 1000 * self.sfreq2)
        v_hat = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_atoms, n_samples)),
            dims=("run", "sensors", "atom", "atom_time"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "atom": np.arange(self.n_atoms),
                "atom_time": np.linspace(
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
            dims=("run", "sensors", "alphacsc_component", "mni_coordinates"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "alphacsc_component": np.arange(self.n_atoms),
                "mni_coordinates": np.arange(3)
                },
            name="alphacsc_components_localization")

        alphacsc_components_gof = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types, self.n_atoms)),
            dims=("run", "sensors", "alphacsc_component"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "alphacsc_component": np.arange(self.n_atoms)
                },
            name="alphacsc_components_gof")

        # --------- AlphaCSC events detection --------- #

        alphacsc_detections_timestamps = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "alphacsc_timestamps"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                },
            name="alphacsc_detections_timestamps")

        alphacsc_detections_goodness = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "detections_goodness"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                },
            name="alphacsc_detections_goodness")

        alphacsc_detections_atom = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types,
                      self.n_detected_peaks)),
            dims=("run", "sensors", "detections_atom"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                },
            name="alphacsc_detections_atom")

        # --------- Clusters library --------- #

        alphacsc_components_selected = xr.DataArray(
            np.zeros((self.n_runs, self.n_sensor_types, self.n_atoms)),
            dims=("run", "sensors", "alphacsc_selected"),
            coords={
                "run": np.arange(self.n_runs),
                "sensors": ['grad', 'mag'],
                "alphacsc_selected": np.arange(self.n_atoms)
                },
            name="alphacsc_components_selected")

        clusters_library_timestamps = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("cluster_library_timestamps"),
            name="clusters_library_timestamps")

        clusters_library_atom = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("clusters_library_atom"),
            name="clusters_library_atom")

        clusters_library_sensors = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("clusters_library_sensors"),
            name="clusters_library_sensors")

        clusters_library_run = xr.DataArray(
            np.zeros(self.n_clusters_library_timepoints),
            dims=("clusters_library_run"),
            name="clusters_library_run")

        n_samples = np.int32(self.n_times_cluster_epoch / 1000 * self.sfreq1)
        clusters_mne_localization = xr.DataArray(
            np.zeros((self.n_clusters_library, self.n_fwd_sources,
                      self.n_times_cluster_epoch)),
            dims=("cluster", "fwd_source", "epoch_times"),
            coords={
                "cluster": np.arange(self.n_clusters_library),
                "epoch_times": np.linspace(
                    0, self.n_times_cluster_epoch, n_samples)
                },
            name="clusters_library_mne_localization")

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
            "alphacsc_detections_goodness": alphacsc_detections_goodness,
            "alphacsc_components_selected": alphacsc_components_selected,
            "clusters_library_timestamps": clusters_library_timestamps,
            "clusters_library_atom": clusters_library_atom,
            "clusters_library_sensors": clusters_library_sensors,
            "clusters_library_run": clusters_library_run,
            "clusters_mne_localization": clusters_mne_localization,
            "iz_predictions": iz_predictions
            })
        return ds
