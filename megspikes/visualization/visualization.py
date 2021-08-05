import matplotlib.pylab as plt
import mne
import numpy as np
import xarray as xr
from scipy import signal
import holoviews as hv
# from holoviews import opts
import hvplot.xarray
import panel.widgets as pnw
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

    def plot_ica_peaks_detections(self, ds, sel_pipe, filter_ica=False):
        slider = pnw.RangeSlider(
            name='time', start=0., end=10., value=(0., 5.), step=0.01)

        ica_src = ds.ica_sources.copy(deep=True)
        if filter_ica:
            sfreq = 200
            freq = np.array([20, 90]) / (sfreq / 2.0)
            b, a = signal.butter(3, freq, "pass")
            ica_src.values = signal.filtfilt(b, a, ica_src.values)

        sel_detections = dict(
            detection_property='detection', pipeline=sel_pipe)
        detections = ica_src * ds.detection_properties.loc[sel_detections]
        detections.name = 'ica_peak_detection'

        sel_ica_component = dict(
            detection_property='ica_component', pipeline=sel_pipe)
        ica_source_ind = ds.detection_properties.loc[sel_ica_component]

        for sens in ds.sensors.values:
            for ica_comp_ind in ds.ica_component.values:
                mask = ica_source_ind.loc[sens] != ica_comp_ind
                detections.loc[sens, ica_comp_ind][mask] *= 0

        # by='ica_component', subplots=True
        # ls = hv.link_selections.instance()
        # # plot =  ls(src) * ls(peaks)

        detections = detections.where(detections != 0)
        ds_small = xr.merge([ica_src, detections])

        interactive_ds = ds_small.interactive().sel(time=slider)

        src = (interactive_ds.ica_sources
               .hvplot(kind='line', width=800))
        # , subplots=True, group='ica_component'
        peaks = (interactive_ds.ica_peak_detection
                 .hvplot(kind='scatter', color='r'))
        # peaks_properties = (
        # interactive_ds
        # .ica_peak_detection.hvplot(kind='table'))
        return src * peaks  # + peaks_properties

    # ica_sources_peaks = plot_ica_peaks_detections(ds, 'aspire_alphacsc_run_3', False)
    # pn.Column(ica_sources_peaks[0], ica_sources_peaks[1])
    # ica_sources_peaks

    # def plot_ica_sources_and_peaks(self, results):
    #     def ica_ts(run, sens, filt, t, window, scale):
    #         sfreq = 200.
    #         sources = results.ica_sources.values[sens]
    #         t_max = min(int((t + window/2)*sfreq), sources.shape[1])
    #         t_min = max(int((t - window/2)*sfreq), 0)
    #         all_sources = []
    #         first_sample = sources[:, :t_min].shape[1]
    #         sources = sources[:, t_min:t_max]*scale
    #         for n, i in enumerate(sources):
    #             peaks = results.ica_peaks_timestamps.values[
    #                 run, sens, :].copy()
    #             peak_source = results.ica_peaks_sources.values[
    #                 run, sens, :].copy()
    #             mask = (peaks < t_max) & (peaks > t_min) & (peaks != 0)
    #             peaks = peaks[mask]
    #             peak_source = peak_source[mask]
    #             # peak_selected = results.ica_peaks_selected.loc[
    #             # run, sensors, :].values.copy()
    #             if filt == 1:
    #                 freq = np.array([20, 90]) / (sfreq / 2.0)
    #                 b, a = signal.butter(3, freq, "pass")
    #                 i = signal.filtfilt(b, a, i)
    #             curve = hv.Curve(i, 'time', 'ica')
    #             peaks_n = peaks[peak_source == n] - first_sample
    #             scatter = hv.Scatter((peaks_n, i[np.int32(peaks_n)]),
    #                                  'time', 'ica')
    #             all_sources.append(curve*scatter)
    #     dmap = hv.DynamicMap(
    #         ica_ts, kdims=['run', 'sens', 'filt', 't', 'window', 'scale'])
    #     return dmap.redim.range(run=(0, 3), sens=(0, 1), filt=(0, 1),
    #                             t=(2, 10), window=(3, 9), scale=(1, 10))

    # layout = hv.Layout(all_sources).cols(1).opts(
    #     opts.Layout(), # shared_axes=False
    #     opts.Curve(line_width=0.5,
    #                 color='k', yaxis=None, height=100, line_alpha=0.6,
    # width=800,
    #                 xaxis=None),
    #     opts.Scatter(color='indianred'))
    # return layout
    #     sfreq = results.ica_sources.attrs['sfreq']
    #     assert sfreq == results.ica_peaks_timestamps.attrs['sfreq']
    #     holomap = hv.HoloMap(kdims=[
    #         'run', 'sensors', 'filtering', 'ica_source'])
    #     for run in [0, 1, 2, 3]:
    #         for sens in ['grad', 'mag']:
    #             for filt in ['raw', 'filtered']:
    #                 sources = results.ica_sources.loc[sens].values
    #                 for n, i in enumerate(sources):
    #                     peaks = results.ica_peaks_timestamps.loc[
    #                         run, sens, :].values.copy()
    #                     peak_source = results.ica_peaks_sources.loc[
    #                         run, sens, :].values.copy()
    #                     # peak_selected = results.ica_peaks_selected.loc[
    #                     # run, sensors, :].values.copy()
    #                     if filt == 'filtered':
    #                         freq = np.array([20, 90]) / (sfreq / 2.0)
    #                         b, a = signal.butter(3, freq, "pass")
    #                         i = signal.filtfilt(b, a, i)
    #                     peak_source = peak_source[peaks != 0]
    #                     peaks = peaks[peaks != 0]
    #                     curve = hv.Curve(i, 'time', 'ica')
    #                     peaks_n = peaks[peak_source == n]
    #                     scatter = hv.Scatter(
    #                         (peaks_n, i[np.int32(peaks_n)]), 'time', 'ica')
    #                     holomap[run, sens, filt, n] = curve*scatter
    #     layout = holomap.layout('ica_source').cols(1).opts(
    #         opts.Layout(shared_axes=False),
    #         opts.Curve(width=800, height=100, line_width=0.5,
    #                    line_color='k', line_alpha=0.6, yaxis=None,
    #                    xaxis=None),
    #         opts.Scatter(color='indianred'))
    #     return layout

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
