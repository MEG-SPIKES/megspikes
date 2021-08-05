import matplotlib.pylab as plt
import mne
import numpy as np
import xarray as xr
from scipy import signal
import holoviews as hv
# from holoviews import opts
# import hvplot.xarray
from sklearn import preprocessing
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
                                    sel_pipe: str = 'aspire_alphacsc_run_1',
                                    filter_ica: bool = True):
        def select_detections(ts_for_overlay, sel_pipe,
                              detection_property='detection',
                              by='ica_component',
                              name='ica_peak_detection'):

            detections = xr.zeros_like(ts_for_overlay)
            detections.name = name

            sel_ica_component = dict(detection_property=by,
                                     pipeline=sel_pipe)
            ica_source_ind = ds.detection_properties.loc[sel_ica_component]
            for sens in ds.sensors.values:
                for ica_comp_ind in ds.ica_component.values:
                    sel_detections = dict(
                        detection_property=detection_property,
                        pipeline=sel_pipe,
                        sensors=sens)
                    detections.loc[sens, ica_comp_ind] = (
                        ts_for_overlay.loc[sens, ica_comp_ind] *
                        ds.detection_properties.loc[sel_detections])
                    mask = ica_source_ind.loc[sens] != ica_comp_ind
                    detections.loc[sens, ica_comp_ind][mask] *= 0

            detections = detections.where(detections != 0)
            return detections

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
            for i in range(ica_src2.values.shape[0]):
                ica_src2.values[i, :] = signal.filtfilt(
                    b, a, ica_src2.values[i, :])
                ica_src2.values[i, :] = preprocessing.robust_scale(
                    ica_src2.values[i, :])

        detections = select_detections(
            ica_src, sel_pipe, name='ica_peak_detection')

        detections_filt = select_detections(
            ica_src2, sel_pipe, name='ica_peak_detection_filt')

        detections_cleaned = select_detections(
            ica_src, sel_pipe, detection_property='selected_for_alphacsc',
            name='selected_for_alphacsc')

        detections_cleaned_filt = select_detections(
            ica_src2, sel_pipe, detection_property='selected_for_alphacsc',
            name='selected_for_alphacsc_filt')

        ds_plot = xr.merge([
            ica_src, ica_src2, detections, detections_filt,
            ds.detection_properties.sel(pipeline=sel_pipe),
            detections_cleaned, detections_cleaned_filt])
        ds_plot = ds_plot.interactive().sel(time=time_slider).sel(
            sensors=select_sensors)

        plot_src = (
            ds_plot.original_ica_sources
            .hvplot(kind='line', width=1000, height=100,
                    yaxis=None, xaxis=None, subplots=True, color='k',
                    line_width=0.5, by=['ica_component']).cols(1))

        plot_src_filt = (
            ds_plot.filtered_ica_sources
            .hvplot(kind='line', width=1000, height=100,
                    yaxis=None, xaxis=None, subplots=True, color='k',
                    line_width=0.5, by=['ica_component']).cols(1))

        kw_params = dict(width=1000, height=100,
                         tools=['hover', 'lasso_select', 'tap', 'box_select'],
                         yaxis=None, xaxis=None, subplots=True, kind='scatter',
                         by=['ica_component'])

        plot_det = (
            ds_plot.ica_peak_detection
            .hvplot(color='b', alpha=0.2, **kw_params).cols(1))

        plot_det_filt = (
            ds_plot.ica_peak_detection_filt
            .hvplot(color='b', alpha=0.2, **kw_params).cols(1))

        plot_selected_det = (
            ds_plot.selected_for_alphacsc
            .hvplot(color='r', alpha=0.5, **kw_params).cols(1))

        plot_selected_det_filt = (
            ds_plot.selected_for_alphacsc_filt
            .hvplot(color='r', alpha=0.5, **kw_params).cols(1))

        table = (
            ds_plot.detection_properties
            .hvplot(kind='table', y='detection_property', x='time', width=1000,
                    by=['ica_component']))
        ts_det = plot_src*plot_det*plot_selected_det
        ts_det_filt = plot_src_filt*plot_det_filt*plot_selected_det_filt
        full_plot = (ts_det + ts_det_filt + table).cols(1)
        return full_plot

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
