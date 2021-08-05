import matplotlib.pylab as plt
import mne
import numpy as np
import xarray as xr
from scipy import signal
import holoviews as hv
# from holoviews import opts
import hvplot.xarray
import panel.widgets as pnw
import panel as pn
hv.extension('bokeh')

mne.set_log_level("WARNING")


class PlotPipeline():
    def __init__(self) -> None:
        pass

    def plot_ica_components(self, arr: xr.DataArray, info: mne.Info,
                            sensors: str = 'grad', n_columns: int = 5):
        """Plot ICA components.
        NOTE: the colorbar is not the same for all components

        Parameters
        ----------
        arr : xr.DataArray
            DataArray from the results DataSet with the shape ica_components
            by channels
        info : mne.Info
            info data structure for plotting. raw_fif.info file could be used
        sensors : str, optional
            sensors for plotting, by default 'grad'
        n_columns : int, optional
            number of columns in the plot, by default 5

        Returns
        -------
        matplotlib.figure.Figure
            Plot with all ica components
        """
        info = mne.pick_info(info, mne.pick_types(info, meg=sensors))
        data = arr.loc[sensors, :, :].values
        n_sens = arr.attrs[f"n_{sensors}"]

        # set figure
        n_components = data.shape[0]
        n_rows = n_components // n_columns
        if n_rows < n_components / n_columns:
            n_rows += 1
        figsize = (4 * n_columns, 3 * n_rows)
        fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize)

        for k, ax in enumerate(axes.flatten()):
            if k < n_components:
                #  vmin=data.min(), vmax=data.max(),
                mne.viz.plot_topomap(
                    data[k, :n_sens], info, axes=ax, show=False)
                ax.set(title="Spatial pattern {}".format(k))
            else:
                ax.axis('off')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        return fig

    def plot_sources_and_detections(self, ds: xr.Dataset,
                                    sel_pipe: str = 'aspire_alphacsc_run_3',
                                    filter_ica: bool = True):

        select_sensors = pn.widgets.Select(
            name='Sensors', options=['grad', 'mag'])

        time_slider = pnw.RangeSlider(
            name='time', start=0., end=float(ds.time.max().values),
            value=(0., 5.), step=0.01)

        ica_src = ds.ica_sources.copy(deep=True)
        ica_src.name = 'original_ica_sources'

        ica_src2 = ds.ica_sources.copy(deep=True)
        ica_src2.name = 'filtered_ica_sources'

        if filter_ica:
            sfreq = 200
            freq = np.array([20, 90]) / (sfreq / 2.0)
            b, a = signal.butter(3, freq, "pass")
            ica_src2.values = signal.filtfilt(b, a, ica_src2.values)

        sel_detections = dict(detection_property='detection',
                              pipeline=sel_pipe)
        detections = ica_src * ds.detection_properties.loc[sel_detections]
        detections.name = 'ica_peak_detection'

        sel_ica_component = dict(detection_property='ica_component',
                                 pipeline=sel_pipe)
        ica_source_ind = ds.detection_properties.loc[sel_ica_component]    
        for sens in ds.sensors.values:
            for ica_comp_ind in ds.ica_component.values:
                mask = ica_source_ind.loc[sens] != ica_comp_ind
                detections.loc[sens, ica_comp_ind][mask] *= 0
        detections = detections.where(detections != 0)

        subcorrs = ds.detection_properties.loc[
            dict(detection_property='detection',
                 pipeline=sel_pipe)].copy(deep=True)
        subcorrs.name = 'ica_peaks_subcorrs'

        ds_plot = xr.merge([ica_src, ica_src2, detections, subcorrs])
        ds_plot = ds_plot.interactive().sel(time=time_slider).sel(
            sensors=select_sensors)

        plot_src = (
            ds_plot.original_ica_sources
            .hvplot(kind='line', width=1000, height=100,
                    yaxis=None, xaxis=None, subplots=True,
                    by=['ica_component']).cols(1))

        plot_src_filt = (
            ds_plot.filtered_ica_sources
            .hvplot(kind='line', width=1000, height=100,
                    yaxis=None, xaxis=None, subplots=True,
                    by=['ica_component']).cols(1))

        plot_det = (
            ds_plot.ica_peak_detection
            .hvplot(kind='scatter', color='r', width=1000, height=100,
                    yaxis=None, xaxis=None, subplots=True,
                    by=['ica_component']).cols(1))

        table = (
            ds_plot.ica_peaks_subcorrs
            .hvplot(kind='table', y='detection_property', x='time', width=1000,
                    by=['ica_component']))
        return (plot_src*plot_det + plot_src_filt + table).cols(1)

    def plot_ica_peaks_localizations(self):
        pass

    def plot_aspire_clusters(self):
        pass

    def plot_alphacsc_atoms(self):
        pass

    def plot_clusters_library(self):
        pass

    def plot_iz_prediction(self):
        pass
