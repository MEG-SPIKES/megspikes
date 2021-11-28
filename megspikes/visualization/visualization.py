import logging

import matplotlib.pylab as plt
import mne
import numpy as np
import panel as pn
import param
import xarray as xr
from nilearn import plotting
from scipy import signal
from sklearn import preprocessing

from ..casemanager.casemanager import CaseManager
from ..database.database import (check_and_read_from_dataset,
                                 check_and_write_to_dataset)
from ..localization.localization import Localization, PredictIZClusters
from ..utils import (create_epochs, spike_snr_all_channels,
                     spike_snr_max_channel)

# NOTE: pn.extension('tabulator') should be before hvplot.xarray
pn.extension('tabulator')
import hvplot.xarray  # noqa

mne.set_log_level("WARNING")


class PlotDetections(Localization):
    """Plot spikes' detection results.
    """
    def __init__(self, ds: xr.Dataset, case: CaseManager) -> None:
        self.ds = ds.copy(deep=True)
        self.setup_fwd(case, sensors=True, spacing='oct5')
        self.grad = self.ds.attrs['grad']
        self.mag = self.ds.attrs['mag']
        self.dprop = self.ds.detection_properties.copy(deep=True)
        self.sica = self.ds.ica_sources.copy(deep=True)
        self.sfreq = self.ds.time.attrs['sfreq']
        self.atom_prop = self._prepare_atoms_properties()

    @property
    def dataset(self):
        return self.ds

    @property
    def forward_model(self):
        return self.fwd

    @property
    def alphacsc_atoms(self):
        return self.atom_prop

    def _prepare_atoms_properties(self):
        atoms_properties = self.ds.alphacsc_atoms_properties.to_dataframe()
        atoms_properties = atoms_properties.reset_index().pivot(
            index=['run', 'sensors', 'alphacsc_atom'],
            columns='alphacsc_atom_property',
            values='alphacsc_atoms_properties')
        return atoms_properties


class DetectionsViewer(param.Parameterized):
    run = param.Selector(default=0, label="Run")
    ica_comp = param.Selector(default=0, label="ICA component")
    sensors = param.Selector(default='grad', objects=['mag', 'grad'],
                             label="Sensors")
    atom = param.Selector(default=0, label="Atom")
    detection_type = param.Selector(
        default='ica_detection',
        objects=['ica_detection', 'selected_for_alphacsc',
                 'alphacsc_detection'],
        label="Detection type")
    preprocess_ica_ts = param.Boolean(False, label="Preprocess ICA timeseries")
    time = param.Range(default=(0, 5))

    def __init__(self, ds: xr.Dataset, case: CaseManager, **params) -> None:
        super().__init__(**params)
        self.data = PlotDetections(ds, case)
        self.param.run.objects = self.data.ds.run.values
        self.param.atom.objects = self.data.ds.alphacsc_atom.values
        self.ts_type = 'ica_component'
        self.param.time.bounds = (0, self.data.ds.time.values[-1])

    # --------------------------- ICA components --------------------------- #
    def view_ica(self):
        """View ICA components"""
        app = pn.Column(
            pn.Param(
                self.param,
                parameters=['sensors'],
                default_layout=pn.Row,
                name="Select",
                width=800
                ),
            pn.Row(
                self._plot_ica_components,
                width=1000,
                height=600,
                scroll=True))
        return app

    @param.depends('sensors')
    def _plot_ica_components(self, n_columns: int = 3):
        """Plot ICA components.
        NOTE: the colorbar is not the same for all components

        Parameters
        ----------
        n_columns : int, optional
            number of columns in the plot, by default 5
        """
        info = mne.pick_info(self.data.info, mne.pick_types(
            self.data.info, meg=self.sensors))
        data = self.data.ds.ica_components.loc[
            dict(channel=self.data.ds.channel_names.attrs[
                self.sensors])].values
        n_sens = len(self.data.ds.channel_names.attrs[self.sensors])

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
        plt.close()
        return pn.pane.Matplotlib(fig, tight=True)

    # --------------------------- ICA sources --------------------------- #

    def view_ica_sources_and_peaks(self):
        self._prepare_ica_sources()
        self._prepare_detections_overlay()
        app = pn.Column(
            '### Plot detections on ICA sources time-series',
            pn.Param(
                self.param,
                parameters=['sensors', 'run', 'ts_type', 'detection_type'],
                default_layout=pn.Row,
                name="Parameters",
                width=800
                ),
            pn.Param(
                self.param,
                parameters=['time', 'preprocess_ica_ts'],
                default_layout=pn.Row,
                name="Actions",
                width=800),
            pn.Row(
                self._plot_ica_sources_with_overlay, width=1000, height=600,
                scroll=True)
            )
        return app

    @param.depends('sensors', 'preprocess_ica_ts', watch=True)
    def _prepare_ica_sources(self):
        self.ica_ts = self.data.sica.sel(
            sensors=self.sensors).copy(deep=True)
        if self.preprocess_ica_ts:
            self.ica_ts = self._filter_ica_sources(self.ica_ts)

    @param.depends('run', 'sensors', 'detection_type', 'preprocess_ica_ts',
                   watch=True)
    def _prepare_detections_overlay(self):
        detections = xr.zeros_like(self.ica_ts)
        detections.name = "detections_for_overlay"
        sel1 = dict(detection_property=self.ts_type, run=self.run,
                    sensors=self.sensors)
        sel2 = dict(detection_property=self.detection_type, run=self.run,
                    sensors=self.sensors)

        ica_source_ind = self.data.dprop.loc[sel1]
        if self.detection_type == 'alphacsc_detection':
            alignment = self.data.dprop.sel(
                detection_property='ica_alphacsc_aligned', run=self.run,
                sensors=self.sensors).values
            ind2 = np.where(alignment != 0)[0]  # alpha peak
            ind1 = np.int32(alignment[ind2])  # ica peak

        for ica_comp_ind in self.data.ds.ica_component.values:
            detections.loc[ica_comp_ind] = (
                self.ica_ts.loc[ica_comp_ind] * self.data.dprop.loc[sel2])
            mask = ica_source_ind != ica_comp_ind
            if self.detection_type == 'alphacsc_detection':
                mask2 = np.zeros_like(mask, dtype=bool)
                mask2[ind2] = mask[ind1]
                detections.loc[ica_comp_ind][mask2] *= 0

            else:
                detections.loc[ica_comp_ind][mask] *= 0
        self.ica_source_ind = ica_source_ind
        self.detections_overlay = detections.where(detections != 0)

    @param.depends('run', 'sensors', 'detection_type', 'preprocess_ica_ts',
                   'time')
    def _plot_ica_sources_with_overlay(self):
        time_slice = slice(self.time[0], self.time[1])
        # .loc[self.ica_source_ind]
        plot_ts = self.ica_ts.loc[:, time_slice].hvplot(
            kind='line', width=800, height=100,
            yaxis=None, xaxis=None, subplots=True, color='k',
            line_width=0.5, by=['ica_component']).cols(1)

        plot_overlay = self.detections_overlay.loc[:, time_slice].hvplot(
            color='r', alpha=0.5, width=800, height=100,
            # tools=['hover', 'lasso_select', 'tap', 'box_select'],
            yaxis=None, xaxis=None, subplots=True, kind='scatter',
            by=['ica_component']).cols(1)
        return plot_ts * plot_overlay

    def _filter_ica_sources(self, ts, sfreq: float = 200.):
        sfreq = 200
        freq = np.array([20, 90]) / (sfreq / 2.0)
        b, a = signal.butter(3, freq, "pass")
        for i in range(ts.values.shape[0]):
            ts.values[i, :] = signal.filtfilt(
                b, a, ts.values[i, :])
            ts.values[i, :] = preprocessing.robust_scale(
                ts.values[i, :])
        return ts

    # ------------------- ICA peaks spacial clustering --------------------- #
    def view_ica_peak_localizations(self):
        app = pn.Column(
            pn.Param(
                self.param,
                parameters=['sensors', 'run', 'detection_type'],
                default_layout=pn.Row,
                name="Select",
                width=800
                ),
            pn.Row(
                self._plot_ica_peak_localizations,
                width=1000,
                height=400,
                scroll=True))
        return app

    @param.depends('sensors', 'run', 'detection_type')
    def _plot_ica_peak_localizations(self):
        sel_x_mni = dict(detection_property='mni_x', run=self.run,
                         sensors=self.sensors)
        x_mni = self.data.dprop.sel(sel_x_mni).values
        sel_y_mni = dict(detection_property='mni_y', run=self.run,
                         sensors=self.sensors)
        y_mni = self.data.dprop.sel(sel_y_mni).values
        sel_z_mni = dict(detection_property='mni_z', run=self.run,
                         sensors=self.sensors)
        z_mni = self.data.dprop.sel(sel_z_mni).values
        det_type = self.detection_type
        if self.detection_type == 'alphacsc_detection':
            det_type = 'selected_for_alphacsc'
        self_detections = dict(detection_property=det_type,
                               run=self.run, sensors=self.sensors)
        detections = self.data.dprop.sel(self_detections).values

        markers = np.vstack([x_mni, y_mni, z_mni]).T[detections != 0]
        fig, ax = plt.subplots(figsize=(12, 7))
        display = plotting.plot_glass_brain(
                    None, display_mode='lzry', figure=fig, axes=ax)
        display.add_markers(markers, marker_color='tomato', alpha=0.2)

        plt.close()
        return pn.pane.Matplotlib(fig, tight=True)

    # -------------------------- AlphaCSC atoms ---------------------------- #
    def view_alphacsc_atoms(self):
        """View AlphaCSC atoms"""
        app = pn.Column(
            pn.Param(
                self.param,
                parameters=['sensors', 'run'],
                default_layout=pn.Row,
                name="Select",
                width=800
                ),
            pn.Row(
                self._plot_alphacsc_atoms,
                width=1000,
                height=600,
                scroll=True))
        return app

    @param.depends('sensors', 'run')
    def _plot_alphacsc_atoms(self):
        ds = self.data.ds.sel(run=self.run, sensors=self.sensors)
        info = mne.pick_info(self.data.info, mne.pick_types(
            self.data.info, meg=self.sensors))
        u_hat = check_and_read_from_dataset(ds, 'alphacsc_u_hat')
        u_hat = u_hat[:, self.data.ds.channel_names.attrs[self.sensors]]
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
            t = ds.atom_v_time.values
            ax.plot(t, v_k)
            # ax.set_xlim(0, int(round(sfreq * atom_width)) / sfreq)
            ax.set(xlabel='Time (sec)',
                   title="Temporal pattern %d" % kk)

        fig.tight_layout()
        plt.close()
        return pn.pane.Matplotlib(fig, tight=True)

    # --------------------------- AlphaCSC clusters ------------------------- #

    def view_alphacsc_clusters(self, raw: mne.io.Raw):
        self.raw = raw
        app = pn.Column(
            pn.Param(
                self.param,
                parameters=['sensors', 'run', 'atom'],
                default_layout=pn.Row,
                name="Select",
                width=800
                ),
            pn.Row(
                self._plot_alphacsc_clusters,
                width=1000,
                height=400,
                scroll=True))
        return app

    @param.depends('sensors', 'run', 'atom')
    def _plot_alphacsc_clusters(self):
        atom = self.atom
        ds = self.data.ds.sel(run=self.run, sensors=self.sensors)

        sfreq = self.raw.info['sfreq']
        detections = check_and_read_from_dataset(
            ds, 'detection_properties',
            dict(detection_property='alphacsc_detection'))
        atoms = check_and_read_from_dataset(
            ds, 'detection_properties',
            dict(detection_property='alphacsc_atom'))
        goodness = check_and_read_from_dataset(
            ds, 'alphacsc_atoms_properties',
            dict(alphacsc_atom_property='goodness'))
        u_hat = check_and_read_from_dataset(ds, 'alphacsc_u_hat')
        u_hat = u_hat[:, self.data.ds.channel_names.attrs[self.sensors]]
        v_hat = check_and_read_from_dataset(ds, 'alphacsc_v_hat')
        goodness = goodness[atom]
        u_hat = u_hat[atom]
        max_channel = np.argmax(u_hat)
        v_hat = v_hat[atom]
        v_hat = v_hat / (np.max(np.abs(v_hat)))
        v_hat_times = np.linspace(-0.25, 0.25, len(v_hat))
        # v_hat_times = ds.atom_v_time.values
        # v_hat_times -= v_hat_times.mean()

        detection_mask = (detections > 0) & (atoms == atom)
        spikes = np.where(detection_mask)[0]
        spikes = (spikes / self.data.sfreq) * sfreq
        spikes += self.raw.first_samp
        epochs = create_epochs(self.raw, spikes, -0.25, 0.25,
                               sensors=self.sensors)
        n_samples_epoch = len(epochs.times)
        evoked = epochs.average()
        spikes = epochs.get_data()[:, max_channel, :]

        fig = plt.figure(figsize=(10, 5), dpi=150)
        ax1 = plt.subplot(2, 2, 1)
        spikes_max_channel = spikes.T/(np.max(np.abs(spikes)))
        spikes_max_channel_times = np.linspace(
            -0.25, 0.25, n_samples_epoch)
        ax1.plot(spikes_max_channel_times, spikes_max_channel,
                 lw=0.3, c='k', alpha=0.5, label='Single events')
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
        plt.close()
        return pn.pane.Matplotlib(fig, tight=True)


class PlotClusters(Localization):
    """Plot detected spikes average and localization.
    """
    def __init__(self, ds: xr.Dataset, case: CaseManager) -> None:
        self.ds = ds.copy(deep=True)
        self.setup_fwd(case, sensors=True, spacing='ico5')
        self.prepare_clusters_properties(ds.copy(deep=True))
        self.stc = self.ds.mne_localization.copy(deep=True)
        self.evoked = self.ds.evoked.copy(deep=True)
        self.grad = self.ds.attrs['grad']
        self.mag = self.ds.attrs['mag']

    @property
    def dataset(self):
        return self.ds

    @property
    def forward_model(self):
        return self.fwd

    @property
    def clusters(self):
        return self.clusters_properties

    def prepare_clusters_properties(self, ds: xr.Dataset):
        clusters_properties = ds.cluster_properties.to_dataframe()
        clusters_properties = clusters_properties.reset_index().pivot(
            index='cluster', columns='cluster_property',
            values='cluster_properties')
        self.clusters_properties = clusters_properties
        del self.clusters_properties['cluster_id']
        del self.clusters_properties['atom']
        del self.clusters_properties['pipeline_type']


class ClusterSlopeViewer(param.Parameterized):
    """Clusters slope viewer. """
    cluster = param.Selector(default=0, label="Cluster")
    sensors = param.Selector(default='grad', objects=['mag', 'grad'],
                             label="Sensors")
    timepoint = param.Selector(default='peak', label='Slope timepoint',
                               objects=['baseline', 'slope', 'peak'])
    plot_stc = param.Action(lambda x: x.param.trigger('plot_stc'),
                            label="Plot Cluster Source Estimate")
    rerun_iz_prediction = param.Action(
        lambda x: x.param.trigger('rerun_iz_prediction'),
        label="Rerun IZ prediction")

    plot_iz = param.Action(lambda x: x.param.trigger('plot_iz'),
                           label="Plot IZ prediction")
    plot_evoked = param.Action(lambda x: x.param.trigger('plot_evoked'),
                               label="Plot Evoked")
    save_ds = param.Action(lambda x: x.param.trigger('save_ds'),
                           label="Save Dataset")
    fname_save_ds = param.String(label='Save dataset path')

    smoothing_steps_one_cluster = param.Integer(
        default=3, bounds=(2, 15), label="One cluster smoothing")
    smoothing_steps_final = param.Integer(
        default=10, bounds=(2, 15), label="Final prediction smoothing")
    amplitude_threshold = param.Number(
        default=0.5, bounds=(0, 1.), label="Amplitude threshold")
    min_sources = param.Integer(
        default=10, bounds=(5, 500), label="Minimum number of sources")
    prediction_is_running = param.Boolean(default=False)

    def __init__(self, ds: xr.Dataset, case: CaseManager, **params):
        super().__init__(**params)
        self.data = PlotClusters(ds, case)
        all_clusters = self.data.clusters_properties.index.values.tolist()
        self.param.cluster.objects = all_clusters
        self.table = pn.widgets.Tabulator(
            self.data.clusters_properties, width=800)
        self.fname_save_ds = str(
            self.data.case.cluster_dataset.with_name(
                f"{self.data.case_name}_clusters_manually_checked.nc"))

    @param.depends('plot_stc', watch=True)
    def _plot_stc_brain(self):
        stc = self.data.array_to_stc(
            self.data.stc.sel(
                cluster=self.cluster, sensors=self.sensors).values,
            self.data.fwd, self.data.case_name)
        self.brain = stc.plot(
            subjects_dir=self.data.freesurfer_dir, hemi='both')
        # pc.brain.widgets['time'].get_value()

    @param.depends('plot_evoked', watch=True)
    def _plot_evoked(self):
        ev = self.data.evoked.sel(cluster=self.cluster).values
        evoked = mne.EvokedArray(ev, self.data.info)
        evoked.plot()

    @param.depends('save_ds', watch=True)
    def _save_dataset(self):
        self._rerun_iz_prediction()
        self.data.ds.to_netcdf(self.fname_save_ds, mode='w', format="NETCDF4",
                               engine="netcdf4")
        logging.warning('DS saved')

    @param.depends('plot_iz', watch=True)
    def _plot_iz_prediciton(self):
        if not self.prediction_is_running:
            stc = self.data.array_to_stc(
                self.data.ds.iz_prediction.sel(
                    iz_prediction_timepoint=self.timepoint).values,
                self.data.fwd, self.data.case_name)
            surfer_kwargs = dict(
                hemi='both',  surface='inflated',  spacing='ico4',
                colorbar=False, background='w', foreground='k',
                colormap='Reds', smoothing_steps=10, alpha=1,
                add_data_kwargs={"fmin": 0, "fmid": 0.5, "fmax": 0.8})
            self.brain = stc.plot(
                subjects_dir=self.data.freesurfer_dir, **surfer_kwargs)
        else:
            logging.warning("IZ prediction is running")

    def _update_dataset(self):
        check_and_write_to_dataset(
            self.data.ds, 'cluster_properties',
            self.data.clusters_properties['time_slope'].values,
            dict(cluster_property='time_slope'))
        check_and_write_to_dataset(
            self.data.ds, 'cluster_properties',
            self.data.clusters_properties['time_baseline'].values,
            dict(cluster_property='time_baseline'))
        check_and_write_to_dataset(
            self.data.ds, 'cluster_properties',
            self.data.clusters_properties['time_peak'].values,
            dict(cluster_property='time_peak'))
        check_and_write_to_dataset(
            self.data.ds, 'cluster_properties',
            self.data.clusters_properties['selected_for_iz_prediction'].values,
            dict(cluster_property='selected_for_iz_prediction'))
        check_and_write_to_dataset(
            self.data.ds, 'cluster_properties',
            self.data.clusters_properties['selected_for_iz_prediction'].values,
            dict(cluster_property='selected_for_iz_prediction'))

    @param.depends('rerun_iz_prediction', watch=True)
    def _rerun_iz_prediction(self):
        self._update_dataset()
        self.prediction_is_running = True
        predict = PredictIZClusters(
            case=self.data.case,
            sensors=True,
            smoothing_steps_one_cluster=self.smoothing_steps_one_cluster,
            smoothing_steps_final=self.smoothing_steps_final,
            amplitude_threshold=self.amplitude_threshold,
            min_sources=self.min_sources)
        self.data.ds, _ = predict.fit_transform((self.data.ds, None))
        self.prediction_is_running = False

    def view(self):
        app = pn.Column(
            pn.Param(
                self.param,
                parameters=['cluster', 'sensors', 'timepoint'],
                # widgets={"cluster": {"widget_type": pn.widgets.Select}}
                # widgets={"plot_stc": {"button_type": "primary"}},
                default_layout=pn.Row,
                name="Select cluster",
                width=800
                ),
            # self.data.clusters_properties,
            self.table,
            pn.Param(
                self.param,
                parameters=['smoothing_steps_final', 'amplitude_threshold',
                            'min_sources', 'rerun_iz_prediction'],
                default_layout=pn.Row,
                name="IZ prediction settings",
                width=800
                ),
            pn.Param(
                self.param,
                parameters=['plot_stc', 'plot_iz', 'plot_evoked', 'save_ds'],
                default_layout=pn.Row,
                name="Actions",
                width=800
                ),
            pn.Param(
                self.param,
                parameters=['fname_save_ds'],
                default_layout=pn.Row,
                name="Information",
                width=800
                )
            )
        return app


def plot_epochs_snr(epochs: mne.Epochs, event_name: str, peak_ind: int = 500,
                    n_max_channels: int = 20):
    data = epochs[event_name].get_data()
    snr_all = spike_snr_all_channels(data, peak_ind)
    snr_max, max_ch = spike_snr_max_channel(data, peak_ind, n_max_channels)

    fig, ax = plt.subplots(1, 2, figsize=(14, 3), dpi=100)
    abs_data = data**2
    ax[0].plot(abs_data.mean(0).T, c='k', linewidth=0.3, alpha=0.5)
    ax[0].plot(abs_data.mean(axis=1).mean(0), c='r')
    ax[0].set_title(f'SNR all channels: {snr_all:.2}dB')
    ax[0].set_xlabel('$Time [ms]$')
    ax[0].set_ylabel('$Amplitude^2$')

    max_chs = data[:, max_ch, :]**2
    ax[1].plot(max_chs.mean(0).T, c='k', linewidth=0.3, alpha=0.5)
    ax[1].plot(max_chs.mean(axis=1).mean(0), c='r')
    ax[1].set_title(f'SNR {n_max_channels} max channels: {snr_max:.2}dB')
    ax[1].set_xlabel('$Time [ms]$')
    ax[1].set_ylabel('$Amplitude^2$')
    plt.suptitle(f"Event {event_name}")
    return fig
