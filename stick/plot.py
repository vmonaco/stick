from .stick import V_t, ms
from .util import decode
import numpy as np
import matplotlib.pyplot as plt


def plot_chronogram(statemon, out=None, plot_spikes=True, subset=None, subplot_labels=None, show_values=None):
    # Show inputs, internal, outputs, each sorted by name
    names = np.array(statemon.neuron_names)
    input_idx = np.array(statemon.inputs, dtype=int)
    output_idx = np.array(statemon.outputs, dtype=int)
    internal_idx = np.array(list(set(np.arange(statemon.n_indices))
                                 .difference(set(statemon.inputs).union(set(statemon.outputs)))), dtype=int)

    if subset is not None and subplot_labels is not None:
        assert len(subset) == len(subplot_labels)

    if len(input_idx) > 0:
        sorted_inputs = input_idx[names[input_idx].argsort()]
    else:
        sorted_inputs = []

    if len(internal_idx) > 0:
        sorted_internal = internal_idx[names[internal_idx].argsort()]
    else:
        sorted_internal = []

    if len(output_idx) > 0:
        sorted_outputs = output_idx[names[output_idx].argsort()]
    else:
        sorted_outputs = []

    if subset == 'top':
        subset = 0

    if subset is None:
        indices = np.r_[sorted_inputs, sorted_internal, sorted_outputs].astype(int)
    elif isinstance(subset, int):
        indices = np.r_[[i for i in np.r_[sorted_inputs, sorted_internal, sorted_outputs].astype(int)
                         if names[i].count(':') <= subset]].astype(int)
    elif subset == 'io':
        indices = np.r_[sorted_inputs, sorted_outputs].astype(int)
    elif isinstance(subset, list) or isinstance(subset, tuple):
        indices = []
        for varname in subset:
            indices.extend([i for i in np.r_[sorted_inputs, sorted_internal, sorted_outputs].astype(int)
                            if varname == names[i]])
        indices = np.r_[indices].astype(int)
    else:
        raise Exception('Unknown neuron subset:', subset)

    if show_values is not None:
        if isinstance(show_values, list) or isinstance(show_values, tuple):
            show_values = {n: 'sequential' for n in show_values}
        else:
            assert isinstance(show_values, dict)

    n_indices = len(indices)
    fig, axes = plt.subplots(n_indices, 1, sharex=True, sharey=True, figsize=(6, 2 * n_indices))
    for plot_idx, (i, ax) in enumerate(zip(indices, axes)):
        neuron_name = statemon.neuron_names[i]
        if subplot_labels is not None:
            neuron_name = subplot_labels[plot_idx]

        if plot_spikes:
            # ax.scatter(statemon.spike_trains[i] / ms, [V_t] * len(statemon.spike_trains[i] / ms), c='r', zorder=2)
            if i in statemon.inputs:
                color = 'blue'
            elif i in statemon.outputs:
                color = 'red'
            else:
                color = 'black'

            for t in statemon.spike_trains[i] / ms:
                ax.arrow(t, 0, 0, V_t * 0.75, color=color, zorder=2, head_width=3.0, head_length=1.0, linewidth=2.0,
                         alpha=0.7)

        ax.plot(statemon.t / ms, statemon.v[i], zorder=1, color='green', alpha=0.7)
        ax.axhline(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--', alpha=0.5, color='k')
        ax.text(0.95, 0.55, neuron_name, ha='right', va='bottom', size=15, transform=ax.transAxes)
        ax.set_yticks([0])
        ax.set_ylim(-(V_t * 1.1), V_t * 1.1)
        ax.tick_params(labelsize=15)

        if show_values is not None and neuron_name in show_values.keys():
            values = decode(statemon.spike_trains[i], method=show_values[neuron_name])

            if show_values[neuron_name] == 'sequential':
                for value, t1, t2 in zip(values, statemon.spike_trains[i][::2] / ms,
                                         statemon.spike_trains[i][1::2] / ms):
                    annot_x = np.mean([t1, t2])
                    annot_y = -2

                    ax.plot([t1, t2], [annot_y, annot_y], color='k', linewidth=1.0)
                    ax.plot([t1, t1], [annot_y, annot_y + 0.5], color='k', linewidth=1.0)
                    ax.plot([t2, t2], [annot_y, annot_y + 0.5], color='k', linewidth=1.0)
                    ax.plot([annot_x, annot_x], [annot_y, annot_y - 0.5], color='k', linewidth=1.0)
                    ax.text(annot_x, annot_y - 1, '%.2f' % value, ha='center', va='top', size=12)

                    # widthB = value * 7
                    # ax.annotate('%.2f' % value, xy=(annot_x, annot_y), xytext=(annot_x, annot_y - 4),
                    #             xycoords='data',
                    #             fontsize=12, ha='center', va='center',
                    #             # bbox=dict(boxstyle='square', fc='white'),
                    #             arrowprops=dict(arrowstyle='-[, widthB=%.4f, lengthB=0.5' % widthB, lw=1.0))

            if show_values[neuron_name] == 'superimposed':
                t1 = statemon.spike_trains[i][0] / ms
                for value_idx, (value, t2) in enumerate(zip(values, statemon.spike_trains[i][1:] / ms)):
                    annot_x = np.mean([t1, t2])
                    annot_y = -0.8 - 3.5 * value_idx

                    ax.plot([t1, t2], [annot_y, annot_y], color='k', linewidth=1.0)
                    ax.plot([t1, t1], [annot_y, annot_y + 0.5], color='k', linewidth=1.0)
                    ax.plot([t2, t2], [annot_y, annot_y + 0.5], color='k', linewidth=1.0)
                    ax.plot([annot_x, annot_x], [annot_y, annot_y - 0.5], color='k', linewidth=1.0)
                    ax.text(annot_x, annot_y - 1, '%.2f' % value, ha='center', va='top', size=12)

                    # widthB = value * 100
                    # ax.annotate('%.2f' % value, xy=(annot_x, annot_y), xytext=(annot_x, annot_y - 2.5),
                    #             xycoords='data',
                    #             fontsize=12, ha='center', va='center',
                    #             # bbox=dict(boxstyle='square', fc='white'),
                    #             arrowprops=dict(
                    #                 arrowstyle='-[, widthB=%.4f, lengthB=2.0' % widthB, lw=1.0,
                    #                 mutation_scale=1.0,
                    #                 mutation_aspect=None)
                    #             )

    axes[0].text(-0.02, 0.95, '$V$', ha='right', va='top', transform=axes[0].transAxes, size=15)
    axes[-1].set_xlabel('$t$ (ms)', size=15)
    plt.tight_layout(h_pad=0.05)

    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight')


def plot_spikes(statemon, out=None, subset=None, subplot_labels=None, show_values=None):
    # Show inputs, internal, outputs, each sorted by name
    names = np.array(statemon.neuron_names)
    input_idx = np.array(statemon.inputs, dtype=int)
    output_idx = np.array(statemon.outputs, dtype=int)
    internal_idx = np.array(list(set(np.arange(statemon.n_indices))
                                 .difference(set(statemon.inputs).union(set(statemon.outputs)))), dtype=int)

    if subset is not None and subplot_labels is not None:
        assert len(subset) == len(subplot_labels)

    if len(input_idx) > 0:
        sorted_inputs = input_idx[names[input_idx].argsort()]
    else:
        sorted_inputs = []

    if len(internal_idx) > 0:
        sorted_internal = internal_idx[names[internal_idx].argsort()]
    else:
        sorted_internal = []

    if len(output_idx) > 0:
        sorted_outputs = output_idx[names[output_idx].argsort()]
    else:
        sorted_outputs = []

    if subset == 'top':
        subset = 0

    if subset is None:
        indices = np.r_[sorted_inputs, sorted_internal, sorted_outputs].astype(int)
    elif isinstance(subset, int):
        indices = np.r_[[i for i in np.r_[sorted_inputs, sorted_internal, sorted_outputs].astype(int)
                         if names[i].count(':') <= subset]].astype(int)
    elif subset == 'io':
        indices = np.r_[sorted_inputs, sorted_outputs].astype(int)
    elif isinstance(subset, list) or isinstance(subset, tuple):
        indices = []
        for varname in subset:
            indices.extend([i for i in np.r_[sorted_inputs, sorted_internal, sorted_outputs].astype(int)
                            if varname == names[i]])
        indices = np.r_[indices].astype(int)
    else:
        raise Exception('Unknown neuron subset:', subset)

    if show_values is not None:
        if isinstance(show_values, list) or isinstance(show_values, tuple):
            show_values = {n: 'sequential' for n in show_values}
        else:
            assert isinstance(show_values, dict)

    n_indices = len(indices)
    fig, ax = plt.subplots(figsize=(6, n_indices/2))
    for plot_idx, i in enumerate(indices):
        neuron_name = statemon.neuron_names[i]

        if subplot_labels is not None:
            neuron_name = subplot_labels[plot_idx]

        if i in statemon.inputs:
            color = 'blue'
        elif i in statemon.outputs:
            color = 'red'
        else:
            color = 'black'

        ax.axhline(plot_idx, color='k', ls='--', alpha=0.25)
        ax.scatter(statemon.spike_trains[i] / ms, [plot_idx]*len(statemon.spike_trains[i]), c=color, zorder=10)

    plt.yticks(np.arange(n_indices), names[indices])
    # ax.set_yticklabels(names[indices])
    ax.set_xlabel('$t$ (ms)', size=15)

    plt.tight_layout(h_pad=0.05)

    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight')
