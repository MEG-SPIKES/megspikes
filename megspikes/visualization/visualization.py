import matplotlib.pylab as plt
import mne
import numpy as np
import xarray as xr
from scipy import signal
import holoviews as hv
# from holoviews import opts
import hvplot.xarray
from sklearn import preprocessing
import panel.widgets as pnw
import panel as pn
from ..utils import create_epochs
from ..database.database import (check_and_read_from_dataset)
hv.extension('bokeh')

mne.set_log_level("WARNING")


class PlotPipeline():
    def __init__(self) -> None:
        pass

    def plot_ica_components(self, ds: xr.Dataset, info: mne.Info,
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
        data = ds.ica_components.loc[
            dict(channel=ds.channel_names.attrs[sensors])].values
        n_sens = len(ds.channel_names.attrs[sensors])

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

    def plot_alphacsc_atoms(self, ds: xr.Dataset, info: mne.Info):
        u_hat = check_and_read_from_dataset(ds, 'alphacsc_u_hat')
        v_hat = check_and_read_from_dataset(ds, 'alphacsc_v_hat')
        plotted_atoms = ds.alphacsc_atom.values

        n_plots = 2  # number of plots by atom
        n_columns = min(3, len(plotted_atoms))
        split = int(np.ceil(len(plotted_atoms) / n_columns))
        figsize = (5 * n_columns, 4 * n_plots * split)
        fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

        for ii, kk in enumerate(plotted_atoms):

            i_row, i_col = ii // n_columns, ii % n_columns
            it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

            # Select the current atom
            u_k = u_hat[kk]
            v_k = v_hat[kk]

            # Plot the spatial map of the atom using mne topomap
            ax = next(it_axes)
            mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
            # _gof = round(gof[ii], 2) if len(gof) != 0 else 0
            # ax.set(title="Spatial pattern {} \n GOF {}".format(kk, _gof))

            # Plot the temporal pattern of the atom
            ax = next(it_axes)
            # t = ds.atom_v_time.values
            ax.plot(v_k)
            # ax.set_xlim(0, int(round(sfreq * atom_width)) / sfreq)
            ax.set(xlabel='Time (sec)',
                   title="Temporal pattern %d" % kk)

        fig.tight_layout()
        return fig

    def plot_alphacsc_clusters(self, ds: xr.Dataset, raw: mne.io.Raw,
                               atom: int = 0):
        sfreq = raw.info['sfreq']
        detections = check_and_read_from_dataset(
            ds, 'detection_properties',
            dict(detection_property='alphacsc_detection'))
        atoms = check_and_read_from_dataset(
            ds, 'detection_properties',
            dict(detection_property='alphacsc_atom'))
        # goodness = check_and_read_from_dataset(
        #     ds, 'alphacsc_atoms_properties',
        #     dict(alphacsc_atom_property='goodness'))
        u_hat = check_and_read_from_dataset(ds, 'alphacsc_u_hat')
        v_hat = check_and_read_from_dataset(ds, 'alphacsc_v_hat')
        goodness = 4  # goodness[atom]
        u_hat = u_hat[atom]
        max_channel = np.argmax(u_hat)
        v_hat = v_hat[atom]
        v_hat = v_hat / (np.max(np.abs(v_hat)))
        v_hat_times = np.linspace(-0.25, 0.25, len(v_hat))
        # v_hat_times = ds.atom_v_time.values

        detection_mask = (detections > 0) & (atoms == atom)
        spikes = np.where(detection_mask)[0]
        spikes = (spikes / ds.time.attrs['sfreq']) * sfreq
        epochs = create_epochs(raw, spikes, -0.25, 0.25)
        n_samples_epoch = len(epochs.times)
        evoked = epochs.average()
        spikes = epochs.get_data()[:, max_channel, :]

        fig = plt.figure(figsize=(15, 7))
        ax1 = plt.subplot(2, 2, 1)
        spikes_max_channel = spikes.T/(np.max(np.abs(spikes)))
        spikes_max_channel_times = np.linspace(
            -0.25, 0.25, n_samples_epoch)
        ax1.plot(spikes_max_channel_times, spikes_max_channel,
                 lw=0.5, c='k', label='Single events')
        ax1.plot(v_hat_times, v_hat,
                 c='r', label='Atom')

        # Clean individual lables
        handles, labels = ax1.get_legend_handles_labels()
        i = 1
        while i < len(labels):
            if labels[i] in labels[:i]:
                del(labels[i])
                del(handles[i])
            else:
                i += 1
        ax1.legend(handles, labels, fontsize='xx-small')
        ax1.set_title("Channel {} and atom {} waveform \n Goodness {}".format(
            epochs.info['ch_names'][max_channel], atom, round(goodness, 2)))

        # Plot epochs image
        ax2 = plt.subplot(2, 2, 2)
        epochs.plot_image(picks=[max_channel], colorbar=False, axes=[ax2],
                          evoked=False, show=False)

        times = [
            epochs.times[t] for t in range(
                10, n_samples_epoch-10, n_samples_epoch // 10)]
        for n, time in enumerate(times):
            ax = plt.subplot(2, len(times), len(times) + n+1)
            evoked.plot_topomap(
                time, axes=ax, show=False, colorbar=False, contours=0)
        return fig



    def plot_clusters_library(self):
        pass

    def plot_iz_prediction(self):
        pass
