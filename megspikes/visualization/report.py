from pathlib import Path

import matplotlib.backends.backend_pdf
import matplotlib.pylab as plt
import mne
import numpy as np
import xarray as xr
from ..database.database import check_and_read_from_dataset
from ..utils import create_epochs, prepare_data
from matplotlib.backends.backend_agg import FigureCanvas


def report_detection(pdf_file_name: Path, dataset: xr.Dataset,
                     data: mne.io.Raw):
    pdf = matplotlib.backends.backend_pdf.PdfPages(str(pdf_file_name))
    data = prepare_data(data.copy(), meg=True, filtering=[2, 90, 50],
                        resample=None, alpha_notch=None)

    for sensors in dataset.sensors.values:
        for run in dataset.run.values:
            fig = plot_ica_components(dataset, data, sensors, run)
            add_image(pdf, fig,
                      f'ICA components \nsensors: {sensors} run: {run}')
            fig = plot_alphacsc_atoms(
                dataset.sel(run=run, sensors=sensors), data, sensors)
            add_image(pdf, fig,
                      f'alphaCSC atoms \nsensors: {sensors} run: {run}')
            for atom in dataset.alphacsc_atom.values:
                fig = plot_alphacsc_clusters(dataset, data, sensors, run, atom)
                add_image(pdf, fig,
                          f'Atom events for sensors: {sensors}, run: {run}, atom: {atom}')
    pdf.close()


def report_atoms_library(pdf_file_name: Path, dataset: xr.Dataset,
                         data: mne.io.Raw):
    pdf = matplotlib.backends.backend_pdf.PdfPages(str(pdf_file_name))
    data = prepare_data(data.copy(), meg=True, filtering=[2, 90, 50],
                        resample=None, alpha_notch=None)

    for sensors in dataset.sensors.values:
        for run in dataset.run.values:
            selected = check_and_read_from_dataset(
                dataset.loc[dict(sensors=sensors, run=run)],
                'alphacsc_atoms_properties',
                dict(alphacsc_atom_property='selected'))
            for atom in np.where(selected != 0)[0]:
                fig = plot_alphacsc_atoms(
                    dataset.loc[
                        dict(sensors=sensors, run=run, alphacsc_atom=[atom])],
                    data, sensors)
                add_image(pdf, fig,
                          f'sensors: {sensors}, run: {run}, atom: {atom}')
                fig = plot_alphacsc_clusters(dataset, data, sensors, run, atom)
                add_image(pdf, fig,
                          f'sensors: {sensors}, run: {run}, atom: {atom}')
    pdf.close()


    # clusters = check_and_read_from_dataset(
    #     dataset,
    #     'alphacsc_atoms_library_properties',
    #     dict(atoms_library_property='library_cluster'), np.int64)
    #
    # atoms = check_and_read_from_dataset(
    #     dataset.loc[dict(sensors=sensors, run=run, atom=atom)],
    #     'alphacsc_atoms_library_properties',
    #     dict(atoms_library_property='library_cluster'), np.int64)
    #
    # detections = check_and_read_from_dataset(
    #     dataset,
    #     'alphacsc_atoms_library_properties',
    #     dict(atoms_library_property='library_detection'), np.int64)
    #
    # sensors = check_and_read_from_dataset(
    #     dataset,
    #     'alphacsc_atoms_library_properties',
    #     dict(atoms_library_property='library_sensors'), np.int64)
    #
    # sensors = check_and_read_from_dataset(
    #     dataset,
    #     'alphacsc_atoms_library_properties',
    #     dict(atoms_library_property='library_sensors'), np.int64)
    #
    # for (cluster, atom, sensor) in zip(clusters, atoms, sensors):
    #     fig = plot_alphacsc_atoms(dataset, data, sensors, run)
    #     add_image(pdf, fig, f'AlphaCSC atoms {sensors} {run}')


def add_image(pdf, fig_plot, title=''):
    """https://matplotlib.org/3.4.3/gallery/misc/agg_buffer_to_array.html"""
    canvas = FigureCanvas(fig_plot)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    fig, ax = plt.subplots(dpi=300)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)
    ax.imshow(img)
    pdf.savefig(fig, orientation='landscape') # bbox_inches='tight',
    plt.close(fig)

# TODO: Use it to plot clusters
# def atoms_library_table(dataset):
#     fig, ax = plt.subplots(dpi=300)
#     # Add a table at the bottom of the axes
#     the_table = plt.table(
#         cellText=dataset.alphacsc_atoms_library_properties.values,
#         colLabels=dataset.atoms_library_property.values)
#     fig.tight_layout()
#     plt.close()
#     return fig

def plot_ica_components(ds: xr.Dataset, raw: mne.io.Raw, sensors: str = 'grad',
                        run: int = 0):
    """Plot ICA components.
    """
    n_columns = 5
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, meg=sensors))
    data = ds.ica_components.loc[
        dict(channel=ds.channel_names.attrs[sensors])].values
    n_sens = len(ds.channel_names.attrs[sensors])

    selected = ds.ica_component_selection.loc[
        dict(sensors=sensors, run=run)].values

    # set figure
    n_components = data.shape[0]
    n_rows = n_components // n_columns
    if n_rows < n_components / n_columns:
        n_rows += 1
    figsize = (4 * n_columns, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize)

    for k, ax in enumerate(axes.flatten()):
        if k < n_components:
            mne.viz.plot_topomap(
                data[k, :n_sens], info, axes=ax, show=False)
            ax.set(title=f"{k} selected: {selected[k] == 1}")
        else:
            ax.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.close()
    return fig


def plot_alphacsc_atoms(ds: xr.Dataset, raw: mne.io.Raw, sensors: str = 'grad'):

    selected = check_and_read_from_dataset(
        ds, 'alphacsc_atoms_properties',
        dict(alphacsc_atom_property='selected'))
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, meg=sensors))
    u_hat = check_and_read_from_dataset(ds, 'alphacsc_u_hat')
    u_hat = u_hat[:, ds.channel_names.attrs[sensors]]
    v_hat = check_and_read_from_dataset(ds, 'alphacsc_v_hat')
    plotted_atoms = ds.alphacsc_atom.values

    n_plots = 2  # number of plots by atom
    n_columns = min(3, len(plotted_atoms))
    split = int(np.ceil(len(plotted_atoms) / n_columns))
    figsize = (5 * n_columns, 4 * n_plots * split)
    fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

    for ii, kk in enumerate(plotted_atoms):

        i_row, i_col = ii // n_columns, ii % n_columns
        if n_columns == 1:
            it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots])
            kk = 0
        else:
            it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

        # Select the current atom
        u_k = u_hat[kk]
        v_k = v_hat[kk]

        # Plot the spatial map of the atom using mne topomap
        ax = next(it_axes)
        mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
        if n_columns != 1:
            ax.set_title(f'Selected: {selected[ii] == 1}')
        else:
            ax.set_title(f'Selected: {selected == 1}')

        # Plot the temporal pattern of the atom
        ax = next(it_axes)
        t = ds.atom_v_time.values
        ax.plot(t, v_k)
        ax.set(xlabel='Time (sec)',
               title="Temporal pattern %d" % kk)

    fig.tight_layout()
    plt.close()
    return fig


def plot_alphacsc_clusters(ds: xr.Dataset, raw: mne.io.Raw,
                           sensors: str = 'grad', run: int = 0, atom: int = 0):
    ds = ds.sel(run=run, sensors=sensors)
    sfreq = raw.info['sfreq']
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
    u_hat = u_hat[:, ds.channel_names.attrs[sensors]]
    v_hat = check_and_read_from_dataset(ds, 'alphacsc_v_hat')
    goodness = goodness[atom]
    u_hat = u_hat[atom]
    max_channel = np.argmax(u_hat)
    v_hat = v_hat[atom]
    v_hat = v_hat / (np.max(np.abs(v_hat)))
    v_hat_times = np.linspace(-0.25, 0.25, len(v_hat))

    detection_mask = (detections > 0) & (atoms == atom)
    spikes = np.where(detection_mask)[0]
    spikes = (spikes / ds.time.attrs['sfreq']) * sfreq
    spikes += raw.first_samp

    # TODO: check if spikes array is empty
    invalid_events_label = ''
    if len(spikes) == 0:
        invalid_events_label = '\n(NOTE: Eevents are invalid)'
        spikes = np.array([300, 700]) + raw.first_samp

    epochs = create_epochs(raw, spikes, -0.25, 0.25, sensors=sensors)
    n_samples_epoch = len(epochs.times)
    evoked = epochs.average()
    spikes = epochs.get_data()[:, max_channel, :]

    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax1 = plt.subplot(2, 2, 1)
    spikes_max_channel = spikes.T/(np.max(np.abs(spikes)))
    spikes_max_channel_times = np.linspace(
        -0.25, 0.25, n_samples_epoch)
    ax1.plot(spikes_max_channel_times, spikes_max_channel,
             lw=0.3, c='k', alpha=0.5,
             label=f'Single events {invalid_events_label}')
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
    return fig
