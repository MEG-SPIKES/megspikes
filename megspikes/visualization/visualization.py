from typing import Tuple

import matplotlib.pylab as plt
import mne

import numpy as np

import xarray as xr

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

    def plot_ica_sources_and_peaks(self, ica_sources: xr.DataArray,
                                   ica_peaks_timestamps: xr.DataArray,
                                   ica_peaks_sources: xr.DataArray,
                                   sensors: str = "grad",
                                   window: Tuple[float] = [0, 20]):
        # IDEA: make the plot interactive using ipywidgets
        # pip install ipympl
        # pip install ipywidgets
        # from ipywidgets import interact, SelectionSlider, Layout
        # interact(update, peak_index=SelectionSlider(
        #     options=np.arange(len(peaks)),value=0, disabled=False,
        #     layout=Layout(width='90%')));
        assert window[1] > window[0]
        sources = ica_sources.loc[sensors, :, :].values
        peaks = ica_peaks_timestamps.loc[0, sensors, :].values
        peaks = np.int32(peaks)
        peaks_sources = ica_peaks_sources.loc[0, sensors, :].values
        peaks_sources = np.int32(peaks_sources)
        unique_sources = np.unique(peaks_sources)
        n_sources = len(unique_sources)

        sfreq = ica_sources.attrs['sfreq']
        assert sfreq == ica_peaks_timestamps.attrs['sfreq']

        fig, axis = plt.subplots(
            n_sources, 1, figsize=(10, n_sources/3 + 2), sharex=True)
        if n_sources == 1:
            axis = [axis]
        time_min, time_max = np.int32(np.round(np.array(window)*sfreq, 0))
        x = np.linspace(window[0], window[1], time_max - time_min)
        for n, (s, ax) in enumerate(zip(unique_sources, axis)):
            print(x.shape)
            ax.plot(x, sources[s, time_min: time_max])
            ax.set_ylabel(f"ICA component {n}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if n == n_sources - 1:
                ax.set_xlabel("Time [s]")
            else:
                ax.get_xaxis().set_visible(False)
                # ax.get_xaxis().set_ticks([])

        for n, ax in enumerate(axis):
            peaks_in_window = (peaks > time_min) & (peaks < time_max)
            for n, peak in enumerate(peaks[peaks < peaks_in_window]):
                ax.scatter(
                    x=peak / sfreq, y=sources[peaks_sources[n], peak], c='r')
        return fig

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
