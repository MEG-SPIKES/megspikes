import matplotlib.pylab as plt
import mne

import xarray as xr

mne.set_log_level("WARNING")


class PlotPipeline():
    def __init__(self) -> None:
        pass

    def plot_ica_components(self, arr: xr.DataArray, info: mne.Info,
                            sensors: str = 'grad', n_columns: int = 5):
        arr = arr.copy()
        info = mne.pick_info(info, mne.pick_types(info, meg=sensors))
        data = arr.loc[sensors, :, :]
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
                    data[k, :n_sens], info, axes=ax,
                    show=False)
                ax.set(title="Spatial pattern {}".format(k))
            else:
                ax.axis('off')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        return fig

    def plot_kurtosis_distribution(self):
        pass

    def plot_ica_peaks_detections(self):
        pass

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
