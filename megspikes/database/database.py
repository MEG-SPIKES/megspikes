from typing import List, Dict, Union, Any, Tuple
from pathlib import Path
import xarray as xr
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import mne


class Database():
    def __init__(self,
                 times: np.ndarray = np.linspace(0, 10, 2000),
                 n_fwd_sources: int = 20_000,
                 sensors: List[str] = ['grad', 'mag'],
                 channels_by_sensors: Dict[str, np.ndarray] = {
                     'grad': np.arange(0, 204),
                     'mag': np.arange(204, 306)},
                 channel_names: List[str] = [f'MEG {i}' for i in range(306)],
                 n_ica_components: int = 20,
                 sfreq1: int = 1000,
                 sfreq2: int = 200,
                 n_aspire_alphacsc_runs: int = 4,
                 n_atoms: int = 3,
                 atom_length: float = 0.5,  # seconds
                 ):
        self.times = times  # seconds
        self.n_fwd_sources = n_fwd_sources
        self.channels_by_sensors = channels_by_sensors
        self.sensors = sensors
        self.channel_names = channel_names
        self.n_ica_components = n_ica_components
        self.sfreq1 = sfreq1  # Hz before downsampling
        self.sfreq2 = sfreq2  # Hz after downsampling
        self.n_atoms = n_atoms
        self.n_aspire_alphacsc_runs = n_aspire_alphacsc_runs
        self.atom_length = atom_length  # ms

    def make_empty_dataset(self) -> xr.Dataset:
        # time coordinate
        sfreq = self.sfreq2
        t = xr.DataArray(
            data=self.times,
            dims=("time"),
            attrs={
                'sfreq': sfreq,
                # 'units': 'seconds'
            },
            name='time')
        meg_data_length = len(t)

        # channels and sensors
        sensors = self.sensors
        channel_names = self.channel_names
        channels = np.arange(len(channel_names))
        grad_inx = self.channels_by_sensors['grad']
        mag_inx = self.channels_by_sensors['mag']

        # pipeline steps
        runs_rng = range(1, self.n_aspire_alphacsc_runs + 1)
        pipelines = [f"aspire_alphacsc_run_{i}" for i in runs_rng]
        pipelines += ["aspire_alphacsc_clusters_library"]
        pipelines += ["manual"]
        self.pipelines_names = pipelines

        # detected evensts properties
        detection_properties = xr.DataArray(
            data=['detection', 'ica_component', 'mni_x', 'mni_y', 'mni_z',
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
            name='detection_properties')

        # ica components
        n_ica_components = self.n_ica_components
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
        n_alphacsc_atoms = self.n_atoms
        alphacsc_atoms_coords = np.arange(n_alphacsc_atoms)
        atom_length = int(self.atom_length * sfreq)
        atom_v_times = np.linspace(
            0, round(atom_length / sfreq, 0), atom_length)
        alphacsc_atoms_properties_coords = xr.DataArray(
            data=['mni_x', 'mni_y', 'mni_z', 'gof', 'goodness'],
            dims=('alphacsc_atom_property'),
            attrs={
                'mni_x_units': 'mm',
                'mni_y_units': 'mm',
                'mni_z_units': 'mm',
                'gof': ('The measurement of the dipole fitting quality.'
                        'Larger values correspond to the better solution'),
                'gof_units': 'percentage between [0, 100]',
                'goodness': 'comprehensive assessment atoms quality'
                },
            name='alphacsc_atoms_properties_coords')

        # vertices = np.arange(20_000)  # vert no
        # sfreq2 = 1000
        # n_clusters = 5
        # mne_length = 1
        # mne_times = np.linspace(0, 1, mne_length*sfreq)

        channel_names = xr.DataArray(
            data=channel_names,
            dims=("channel"),
            coords={"channel": channels},
            attrs={
                'grad': grad_inx,
                'mag': mag_inx,
            },
            name="channel_names")

        detection_properties = xr.DataArray(
            data=np.zeros(
                (len(pipelines), len(sensors),
                 len(detection_properties), meg_data_length)),
            dims=("pipeline", "sensors", "detection_property", "time"),
            coords={
                "pipeline": pipelines,
                "sensors": sensors,
                "detection_property": detection_properties,
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
            data=np.zeros((len(pipelines), len(sensors), n_ica_components)),
            dims=("pipeline", "sensors", "ica_component"),
            coords={
                "pipeline": pipelines,
                "sensors": sensors,
                "ica_component": ica_components_coords,
            },
            name="ica_component_selection")

        alphacsc_z_hat = xr.DataArray(
            data=np.zeros((
                len(pipelines), len(sensors), n_alphacsc_atoms,
                meg_data_length)),
            dims=("pipeline", "sensors", "alphacsc_atom", "time"),
            coords={
                "pipeline": pipelines,
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
                (len(pipelines), len(sensors), n_alphacsc_atoms, atom_length)),
            dims=("pipeline", "sensors", "alphacsc_atom", "atom_v_time"),
            coords={
                "pipeline": pipelines,
                "sensors": sensors,
                "alphacsc_atom": alphacsc_atoms_coords,
                "atom_v_time": atom_v_times
            },
            attrs={
                'units': 'AU'
            },
            name="alphacsc_v_hat")

        alphacsc_u_hat = xr.DataArray(
            data=np.zeros((len(pipelines), n_alphacsc_atoms, len(channels))),
            dims=("pipeline", "alphacsc_atom", "channel"),
            coords={
                "pipeline": pipelines,
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
                (len(pipelines), len(sensors), n_alphacsc_atoms,
                 len(alphacsc_atoms_properties_coords))),
            dims=("pipeline", "sensors", "alphacsc_atom",
                  "alphacsc_atom_property"),
            coords={
                "pipeline": pipelines,
                "sensors": sensors,
                "alphacsc_atom": alphacsc_atoms_coords,
                "alphacsc_atom_property": alphacsc_atoms_properties_coords,
            },
            name="alphacsc_atoms_properties")

        # stc_clusters_library = xr.DataArray(
        #     data=np.zeros((len(vertices), n_alphacsc_atoms, len(mne_times))),
        # )

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
            alphacsc_atoms_properties
            ])
        return ds

    def read_case_info(self, fif_file_path: Union[str, Path],
                       fwd: mne.Forward,
                       sfreq: Union[float, None] = None) -> None:
        if not Path(fif_file_path).is_file():
            raise RuntimeError("Fif file was not found")
        fif_file = mne.io.read_raw_fif(fif_file_path, preload=False)
        if sfreq is not None:
            fif_file.load_data()
            fif_file = fif_file.resample(sfreq, npad="auto")
        info = fif_file.info
        # TODO: other sensors types
        for sens in ['grad', 'mag']:
            self.channels_by_sensors[sens] = mne.pick_types(info, meg=sens)
        self.channel_names = info['ch_names']
        # length of the MEG recording in seconds
        self.times = fif_file.times
        self.n_fwd_sources = sum([len(h['vertno']) for h in fwd['src']])
        del fif_file, fwd

    def select_sensors(self, ds: xr.Dataset, sensors: str,
                       pipeline: str) -> xr.Dataset:
        ds_subset, _ = select_sensors(ds, sensors, pipeline)
        return ds_subset


class LoadDataset(TransformerMixin, BaseEstimator):
    def __init__(self, dataset: Union[str, Path], sensors: str,
                 pipeline: int) -> None:
        self.dataset = dataset
        self.sensors = sensors
        self.pipeline = pipeline

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        ds = xr.load_dataset(self.dataset)
        ds_channels, _ = select_sensors(ds, self.sensors, self.pipeline)
        return (ds_channels, X[1])


class SaveDataset(TransformerMixin, BaseEstimator):
    """Save merge subset of the database in the full dataset
    """
    def __init__(self, dataset: Union[str, Path], sensors: str,
                 pipeline: int) -> None:
        self.dataset = dataset
        self.sensors = sensors
        self.pipeline = pipeline

    def fit(self, X: Tuple[xr.Dataset, Any], y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        ds = xr.load_dataset(self.dataset)
        _, selection = select_sensors(ds, self.sensors, self.pipeline)
        # TODO: avoid manual update
        # Won't working because selection is no applicable to all ds variables
        # ds.loc[selection] = ds_grad
        selection_ch = ds.attrs[self.sensors]
        selection_sens = dict(sensors=self.sensors)
        selection_pipe_sens = dict(
            pipeline=self.pipeline, sensors=self.sensors)
        selection_pipe_ch = dict(
            pipeline=self.pipeline, channel=selection_ch)

        ds['ica_sources'].loc[selection_sens] = X[0].ica_sources
        ds['ica_components'].loc[:, selection_ch] = X[0].ica_components
        ds['ica_component_properties'].loc[
            selection_sens] = X[0].ica_component_properties
        ds['ica_component_selection'].loc[
            selection_pipe_sens] = X[0].ica_component_selection
        ds['detection_properties'].loc[
            selection_pipe_sens] = X[0].detection_properties
        ds['alphacsc_z_hat'].loc[selection_pipe_sens] = X[0].alphacsc_z_hat
        ds['alphacsc_v_hat'].loc[selection_pipe_sens] = X[0].alphacsc_v_hat
        ds['alphacsc_u_hat'].loc[selection_pipe_ch] = X[0].alphacsc_u_hat
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


def select_sensors(ds: xr.Dataset, sensors: str,
                   pipeline: str) -> xr.Dataset:
    channels = ds.channel_names.attrs[sensors]
    selection = dict(pipeline=pipeline, sensors=sensors, channel=channels)
    return ds.loc[selection], selection


def check_and_read_from_dataset(ds: xr.Dataset, da_name: str,
                                selection: Union[Dict[Any, str], None] = None
                                ) -> np.ndarray:
    if not isinstance(da_name, str):
        raise RuntimeError(f"{da_name} has type {type(da_name)}")
    assert da_name in ds.data_vars, (
        f"{da_name} not in dataset")

    if isinstance(selection, dict):
        data = ds[da_name].loc[selection].values.copy()
        assert np.max(data) != np.min(data), (
            f"{da_name}.loc[{selection}] values are all the same")
    else:
        data = ds[da_name].values.copy()
        if da_name != 'ica_component_selection':
            assert np.max(data) != np.min(data), (
                f"{da_name} values are all the same")
        else:
            assert data.any(), (
                f"{da_name} values are all zeros")
    return data


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
