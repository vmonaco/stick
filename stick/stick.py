from brian2 import *

# Which target to build the simulation for
prefs.codegen.target = 'numpy'
# prefs.codegen.target = 'cython'

tau_m = 100 * second
tau_f = 20 * ms
V_r = 0.
V_t = 10.
# threshold: v > V_t - V_t_abstol to avoid numeric errors, mainly due to w_acc
V_t_abstol = 1e-10

T_syn = 1 * ms
T_neu = 10 * us
dt = T_neu  # Half of T_neu. Time step should be less than neuron reaction time

T_min = 10 * ms
T_cod = 100 * ms
T_max = T_min + T_cod

# Equality between decoded values can be tested by isclose(..., atol=abstol)
# The timestep (dt) specifies the minimum distance between two decoded values
# abstol is used to test equality of decoded values, dt for spike times
abstol = dt / T_cod

w_e = V_t
w_i = -w_e
w_acc = V_t * tau_m / T_max
w_delta = V_t / (T_max / dt)

STICK_EQS = '''
dv/dt = (g_e + gate * g_f)/tau_m : 1
g_e : 1
dg_f/dt = -g_f/tau_f : 1
gate : 1
'''

STICK_RESET = '''
v = V_r
g_e = 0
g_f = 0
gate = 0
'''

SYNAPSE_BEHAVIOR = {
    'v': 'v_post += w',
    'g_e': 'g_e_post += w',
    'g_f': 'g_f_post += w',
    'gate': 'gate_post = w'
}


class Neuron:
    def __init__(self):
        self.name = ''
        self.idx = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Neuron(%s)' % self.name


class STICK(object):
    def __init__(self):
        self.sticks = {}
        self.neurons = {}
        self.input_neurons = {}
        self.output_neurons = {}
        self.synapses = dict(zip(SYNAPSE_BEHAVIOR.keys(), [list() for _ in range(len(SYNAPSE_BEHAVIOR))]))

    def create_neuron(self, names):
        if isinstance(names, str):
            return self.add_neuron(names, Neuron())

        neurons = []
        for name in names:
            neurons.append(self.add_neuron(name, Neuron()))
        return tuple(neurons)

    def add_neuron(self, name, neuron):
        if name in self.neurons.keys():
            raise Exception('Already have a Neuron named:', name)

        neuron.name = name
        neuron.idx = len(self.neurons)
        self.neurons[name] = neuron
        return neuron

    def add_stick(self, name, stick):
        if name in self.sticks.keys():
            raise Exception('Already have a STICK named:', name)

        # Add all the neurons and connections from this stick
        for neuron_name, neuron in stick.neurons.items():
            self.add_neuron('%s:%s' % (name, neuron_name), neuron)

        for synapse_type in SYNAPSE_BEHAVIOR.keys():
            self.synapses[synapse_type].extend(stick.synapses[synapse_type])

        self.sticks[name] = stick
        return stick

    def set_input(self, name, neuron):
        assert isinstance(neuron, Neuron)

        if name in self.input_neurons.keys():
            raise Exception('Already have an input named:', name)

        self.input_neurons[name] = neuron

    def set_output(self, name, neuron):
        assert isinstance(neuron, Neuron)

        if name in self.output_neurons.keys():
            raise Exception('Already have an output named:', name)

        self.output_neurons[name] = neuron

    def connect(self, pre, post, synapse_type='v', weight=w_e, delay=T_syn):
        assert isinstance(pre, Neuron) and isinstance(post, Neuron)

        if synapse_type not in self.synapses.keys():
            raise Exception('Unknown synapse type:', synapse_type)

        self.synapses[synapse_type].append((pre, post, weight, delay))

    def __getitem__(self, item):
        if item in self.input_neurons.keys():
            return self.input_neurons[item]

        if item in self.output_neurons.keys():
            return self.output_neurons[item]

        raise Exception('There is no input/output neuron named:', item)

    def run(self, duration, spikes_in={}, return_statemon=False, dt_custom=None):
        # TODO: reset must come before synapses to avoid "dropped" spikes
        # TODO: Should g_e synapses be updated first? Currently take 1 time step to take effect,
        # should have same behavior as a "simulated" linear synapse using only LIF neurons
        Network.schedule = ['start', 'groups', 'thresholds', 'resets', 'synapses', 'end']

        if dt_custom is not None:
            global dt
            global abstol
            dt = dt_custom
            abstol = dt / T_cod

        clock = Clock(dt=dt)
        G = NeuronGroup(len(self.neurons), STICK_EQS, threshold='v > V_t - V_t_abstol', reset=STICK_RESET,
                        method='euler', clock=clock, refractory=0 * ms)

        # TODO: why doesn't this work?
        # for synapse_type in SYNAPSE_TYPES:
        #     if len(self.synapses[synapse_type]) == 0:
        #         continue
        #
        #     synapses = Synapses(G, G, 'w : 1', on_pre=SYNAPSE_BEHAVIOR[synapse_type], clock=clock)
        #     neurons_i, neurons_j, w_ij, delay = zip(*self.synapses[synapse_type])
        #
        #     if any(array(delay) < T_syn / second):
        #         Warning('%s synapse connections contain delay(s) less than T_syn' % synapse_type)
        #
        #     synapses.connect(i=[n.idx for n in neurons_i], j=[n.idx for n in neurons_j])
        #     synapses.w[:] = w_ij
        #     synapses.delay[:] = delay

        if len(self.synapses['v']) > 0:
            v_synapses = Synapses(G, G, 'w : 1', on_pre=SYNAPSE_BEHAVIOR['v'], clock=clock)
            neurons_i, neurons_j, w_ij, delay = zip(*self.synapses['v'])
            v_synapses.connect(i=[n.idx for n in neurons_i], j=[n.idx for n in neurons_j])
            v_synapses.w[:] = w_ij
            v_synapses.pre.delay[:] = delay

        if len(self.synapses['g_e']) > 0:
            g_e_synapses = Synapses(G, G, 'w : 1', on_pre=SYNAPSE_BEHAVIOR['g_e'], clock=clock)
            neurons_i, neurons_j, w_ij, delay = zip(*self.synapses['g_e'])
            g_e_synapses.connect(i=[n.idx for n in neurons_i], j=[n.idx for n in neurons_j])
            g_e_synapses.w[:] = w_ij
            g_e_synapses.delay[:] = delay

        if len(self.synapses['g_f']) > 0:
            g_e_synapses = Synapses(G, G, 'w : 1', on_pre=SYNAPSE_BEHAVIOR['g_f'], clock=clock)
            neurons_i, neurons_j, w_ij, delay = zip(*self.synapses['g_f'])
            g_e_synapses.connect(i=[n.idx for n in neurons_i], j=[n.idx for n in neurons_j])
            g_e_synapses.w[:] = w_ij
            g_e_synapses.delay[:] = delay

        if len(self.synapses['gate']) > 0:
            g_e_synapses = Synapses(G, G, 'w : 1', on_pre=SYNAPSE_BEHAVIOR['gate'], clock=clock)
            neurons_i, neurons_j, w_ij, delay = zip(*self.synapses['gate'])
            g_e_synapses.connect(i=[n.idx for n in neurons_i], j=[n.idx for n in neurons_j])
            g_e_synapses.w[:] = w_ij
            g_e_synapses.delay[:] = delay

        if len(spikes_in) > 0:
            stimuli_indices, input_indices, times = [], [], []
            for i, (name, ti) in enumerate(spikes_in.items()):
                if name not in self.input_neurons.keys():
                    raise Exception('No such input Neuron named:', name)

                input_indices.extend([self.input_neurons[name].idx] * len(ti))
                stimuli_indices.extend([i] * len(ti))
                times.extend(ti)

            inputs = SpikeGeneratorGroup(len(self.input_neurons), stimuli_indices, times, clock=clock)
            input_synapses = Synapses(inputs, G, 'w : 1', on_pre='v_post += w', clock=clock)
            input_synapses.connect(i=stimuli_indices, j=input_indices)
            input_synapses.w[:] = w_e
            input_synapses.delay[:] = 0 * ms

        if return_statemon:
            idx2name = {v.idx: k for k, v in self.neurons.items()}
            statemon = StateMonitor(G, ['v', 'g_e'], record=True)
            statemon.add_attribute('neuron_names')
            statemon.neuron_names = [idx2name[i] for i in range(len(self.neurons))]

        spikemon = SpikeMonitor(G)
        run(duration)

        spikes_out = {}
        for name, neuron in self.output_neurons.items():
            spikes_out[name] = spikemon.t[spikemon.i == neuron.idx]

        if return_statemon:
            statemon.add_attribute('spike_trains')
            statemon.spike_trains = spikemon.spike_trains()
            statemon.add_attribute('inputs')
            statemon.inputs = [neuron.idx for neuron in self.input_neurons.values()]
            statemon.add_attribute('outputs')
            statemon.outputs = [neuron.idx for neuron in self.output_neurons.values()]
            return spikes_out, statemon

        return spikes_out
