from stick import *


def test_busy_beaver():
    # Multiple connections send spikes separated by T_neu = 2*dt
    class BusyBeaver(STICK):
        def __init__(self, V_synapses, g_e_synapses):
            super(BusyBeaver, self).__init__()

            num_neurons = V_synapses.shape[0]
            neurons = self.create_neuron(['n%d' % i for i in range(num_neurons)])
            input_, output_ = self.create_neuron(['input', 'output'])
            self.set_input('input', input_)
            self.set_output('output', output_)

            self.connect(input_, neurons[0])

            for i in range(num_neurons):
                self.connect(neurons[i], output_)

            for i, j in zip(*V_synapses.nonzero()):
                self.connect(neurons[i], neurons[j], weight=V_synapses[i, j], delay=T_syn)

            for i, j in zip(*g_e_synapses.nonzero()):
                self.connect(neurons[i], neurons[j], 'g_e', weight=g_e_synapses[i, j], delay=T_syn)

    # w_acc2 = V_t * tau_m / (20 * ms)
    w_acc2 = w_acc
    V_synapses = np.array([[0., 0.], [0., 0.]])
    g_e_synapses = np.array([[0.5 * w_acc2, w_acc2], [0.5 * w_acc2, -w_acc2]])
    m = BusyBeaver(V_synapses, g_e_synapses)

    spikes_out, statemon = m.run(10000 * ms, spikes_in={
        'input': [0] * ms,
    }, return_statemon=True, dt_custom=1 * ms)

    t = spikes_out['output']
    delta_t = t[1:] - t[:-1]
    values = delta_t / ms
    unique_values = np.unique(values.round().astype(int))
    print(values)
    print(unique_values)
    print(len(unique_values))
    plot_chronogram(statemon)
