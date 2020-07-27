from .stick import *
from .primitives import Iterator


# TODO: spatial router instead of spatial memory? eliminate use of boolean?

class Gate(STICK):
    def __init__(self):
        super(Gate, self).__init__()

        input_, on, off, gate, output = self.create_neuron(['input', 'on', 'off', 'gate', 'output'])

        self.set_input('input', input_)
        self.set_input('on', on)
        self.set_input('off', off)
        self.set_output('output', output)

        self.connect(input_, gate, weight=0.5 * w_e)
        self.connect(input_, gate, weight=0.5 * w_i, delay=T_syn + T_neu)

        self.connect(gate, output)
        self.connect(gate, gate)

        self.connect(on, gate, weight=0.5 * w_e)
        self.connect(off, gate, weight=0.5 * w_i)


class Toggle(STICK):
    def __init__(self):
        super(Toggle, self).__init__()

        input_, switch, init = self.create_neuron(['input', 'switch', 'init'])
        first, last = self.create_neuron(['first', 'last'])
        out0, out1 = self.create_neuron(['out0', 'out1'])

        self.set_input('input', input_)
        self.set_input('switch', switch)

        self.set_output('out0', out0)
        self.set_output('out1', out1)

        self.connect(input_, out0, weight=0.5 * w_e, delay=2 * T_syn + T_neu)
        self.connect(input_, out0, weight=0.5 * w_i, delay=2 * T_syn + 2 * T_neu)
        self.connect(input_, out1, weight=0.5 * w_e, delay=2 * T_syn + T_neu)
        self.connect(input_, out1, weight=0.5 * w_i, delay=2 * T_syn + 2 * T_neu)
        self.connect(input_, init)

        self.connect(init, out0, weight=0.5 * w_e)

        self.connect(switch, first)
        self.connect(switch, last, weight=0.5 * w_e)
        self.connect(switch, init)

        self.connect(first, first, weight=w_i)
        self.connect(first, out0, weight=0.5 * w_i)
        self.connect(first, out1, weight=0.5 * w_e)
        self.connect(first, init, weight=w_i)

        self.connect(last, out0, weight=0.5 * w_e)
        self.connect(last, out1, weight=0.5 * w_i)
        self.connect(last, init, weight=w_i)

        self.connect(out0, out0)
        self.connect(out0, init, weight=w_i)

        self.connect(out1, out1)
        self.connect(out1, init, weight=w_i)


class Toggle2(STICK):
    def __init__(self):
        super(Toggle2, self).__init__()
        N = 2

        iterator = self.add_stick('iterator', Iterator(N))

        input_, next_, init = self.create_neuron(['input', 'switch', 'init'])
        outputs = self.create_neuron(['out%d' % i for i in range(N)])

        self.set_input('input', input_)
        self.set_input('switch', next_)

        self.connect(input_, init, weight=w_e)
        self.connect(input_, init, weight=2 * w_i, delay=T_syn + T_neu)

        self.connect(next_, iterator['next'], weight=w_e)
        self.connect(next_, init, weight=w_e)
        self.connect(next_, init, weight=2 * w_i, delay=T_syn + T_neu)

        self.connect(init, outputs[0], weight=0.5 * w_e)

        for i in range(N):
            self.set_output('out%d' % i, outputs[i])

            self.connect(input_, outputs[i], weight=0.5 * w_e, delay=4 * T_syn + 3 * T_neu)
            self.connect(input_, outputs[i], weight=0.5 * w_i, delay=4 * T_syn + 4 * T_neu)

            self.connect(iterator['out%d' % i], outputs[i], weight=0.5 * w_e)
            self.connect(iterator['out%d' % i], outputs[(i - 1) % N], weight=0.5 * w_i)

            self.connect(outputs[i], outputs[i])


class LinkedRouter(STICK):
    def __init__(self, N):
        super(LinkedRouter, self).__init__()

        iterator = self.add_stick('iterator', Iterator(N))

        inp, next_, init = self.create_neuron(['input', 'next', 'init'])
        outputs = self.create_neuron(['out%d' % i for i in range(N)])

        self.set_input('input', inp)
        self.set_input('next', next_)

        self.connect(next_, iterator['next'], weight=w_e)
        self.connect(next_, init, weight=w_e)
        self.connect(next_, init, weight=w_i, delay=T_syn + T_neu)

        self.connect(init, outputs[-1], weight=0.5 * w_e)

        for i in range(N):
            self.set_output('out%d' % i, outputs[i])

            self.connect(inp, outputs[i], weight=0.5 * w_e, delay=4 * T_syn + 3 * T_neu)
            self.connect(inp, outputs[i], weight=0.5 * w_i, delay=4 * T_syn + 4 * T_neu)

            self.connect(iterator['addr%d' % i], outputs[i], weight=0.5 * w_e)
            self.connect(iterator['addr%d' % i], outputs[(i - 1) % N], weight=0.5 * w_i)

            self.connect(outputs[i], outputs[i])


class LinkedMemRouter(STICK):
    def __init__(self, N):
        super(LinkedMemRouter, self).__init__()

        iterator = self.add_stick('iterator', Iterator(N))

        store_, recall, next_, init = self.create_neuron(['store', 'recall', 'next', 'init'])
        stores = self.create_neuron(['store%d' % i for i in range(N)])
        recalls = self.create_neuron(['recall%d' % i for i in range(N)])

        self.set_input('store', store_)
        self.set_input('recall', recall)
        self.set_input('next', next_)

        self.connect(next_, iterator['next'], weight=w_e)
        self.connect(next_, init, weight=w_e)
        self.connect(next_, init, weight=w_i, delay=T_syn + T_neu)

        self.connect(init, stores[-1], weight=0.5 * w_e)
        self.connect(init, recalls[-1], weight=0.5 * w_e)

        for i in range(N):
            self.set_output('store%d' % i, stores[i])
            self.set_output('recall%d' % i, recalls[i])

            self.connect(store_, stores[i], weight=0.5 * w_e, delay=4 * T_syn + 3 * T_neu)
            self.connect(store_, stores[i], weight=0.5 * w_i, delay=4 * T_syn + 4 * T_neu)

            self.connect(recall, recalls[i], weight=0.5 * w_e, delay=4 * T_syn + 3 * T_neu)
            self.connect(recall, recalls[i], weight=0.5 * w_i, delay=4 * T_syn + 4 * T_neu)

            self.connect(iterator['addr%d' % i], stores[i], weight=0.5 * w_e)
            self.connect(iterator['addr%d' % i], stores[(i - 1) % N], weight=0.5 * w_i)

            self.connect(iterator['addr%d' % i], recalls[i], weight=0.5 * w_e)
            self.connect(iterator['addr%d' % i], recalls[(i - 1) % N], weight=0.5 * w_i)

            self.connect(stores[i], stores[i])
            self.connect(recalls[i], recalls[i])


class DoublyLinkedMemRouter(STICK):
    def __init__(self, N):
        super(DoublyLinkedMemRouter, self).__init__()

        next_, prev_, store_, recall = self.create_neuron(['next', 'prev', 'store', 'recall'])
        n_init, p_init, inhibit = self.create_neuron(['n_init', 'p_init', 'inhibit'])
        nexts = self.create_neuron(['next%d' % i for i in range(N)])
        prevs = self.create_neuron(['prev%d' % i for i in range(N)])
        stores = self.create_neuron(['store%d' % i for i in range(N)])
        recalls = self.create_neuron(['recall%d' % i for i in range(N)])

        self.set_input('next', next_)
        self.set_input('prev', prev_)
        self.set_input('store', store_)
        self.set_input('recall', recall)

        self.connect(next_, n_init)
        self.connect(next_, n_init, weight=w_i, delay=T_syn + T_neu)
        self.connect(next_, p_init, weight=w_i, delay=T_syn + T_neu)
        self.connect(prev_, p_init)
        self.connect(prev_, p_init, weight=w_i, delay=T_syn + T_neu)
        self.connect(prev_, n_init, weight=w_i, delay=T_syn + T_neu)

        self.connect(n_init, nexts[0], weight=0.5 * w_e, delay=dt)
        self.connect(n_init, prevs[-2 % N], weight=0.5 * w_e, delay=dt)
        self.connect(n_init, stores[-1 % N], weight=0.5 * w_e, delay=dt)
        self.connect(n_init, recalls[-1 % N], weight=0.5 * w_e, delay=dt)

        self.connect(p_init, prevs[-1 % N], weight=0.5 * w_e, delay=dt)
        self.connect(p_init, nexts[1 % N], weight=0.5 * w_e, delay=dt)
        self.connect(p_init, stores[0], weight=0.5 * w_e, delay=dt)
        self.connect(p_init, recalls[0], weight=0.5 * w_e, delay=dt)

        for i in range(N):
            self.set_output(stores[i].name, stores[i])
            self.set_output(recalls[i].name, recalls[i])

            # Inputs send +0.5 quickly followed by -0.5. Only "on" neurons will spike
            self.connect(next_, nexts[i], weight=0.5 * w_e, delay=2 * T_syn + T_neu)
            self.connect(next_, nexts[i], weight=0.5 * w_i, delay=2 * T_syn + 2 * T_neu)
            self.connect(prev_, prevs[i], weight=0.5 * w_e, delay=2 * T_syn + T_neu)
            self.connect(prev_, prevs[i], weight=0.5 * w_i, delay=2 * T_syn + 2 * T_neu)

            # Next: chain forward, prev: chain backward
            self.connect(nexts[i], nexts[(i + 1) % N], weight=0.5 * w_e, delay=dt)
            self.connect(prevs[i], prevs[(i - 1) % N], weight=0.5 * w_e, delay=dt)

            # Recurrent connections to reset after spiking (counteracts the 2nd inhibitory synapse from input)
            self.connect(nexts[i], nexts[i], weight=0.5 * w_e, delay=dt)
            self.connect(prevs[i], prevs[i], weight=0.5 * w_e, delay=dt)

            # Next: enable i-1 prev and disable i-2 prev
            self.connect(nexts[i], prevs[(i - 1) % N], weight=0.5 * w_e, delay=dt)
            self.connect(nexts[i], prevs[(i - 2) % N], weight=0.5 * w_i, delay=dt)

            # Prev: enable current next and disable last next
            self.connect(prevs[i], nexts[(i + 1) % N], weight=0.5 * w_e, delay=dt)
            self.connect(prevs[i], nexts[(i + 2) % N], weight=0.5 * w_i, delay=dt)

            # Outputs
            self.connect(store_, stores[i], weight=0.5 * w_e)
            self.connect(store_, stores[i], weight=0.5 * w_i, delay=T_syn + T_neu)
            self.connect(stores[i], stores[i], weight=w_e)

            self.connect(nexts[i], stores[i], weight=0.5 * w_e)
            self.connect(nexts[i], stores[(i - 1) % N], weight=0.5 * w_i)

            self.connect(prevs[i], stores[i], weight=0.5 * w_e)
            self.connect(prevs[i], stores[(i + 1) % N], weight=0.5 * w_i)

            self.connect(recall, recalls[i], weight=0.5 * w_e)
            self.connect(recall, recalls[i], weight=0.5 * w_i, delay=T_syn + T_neu)
            self.connect(recalls[i], recalls[i], weight=w_e)

            self.connect(nexts[i], recalls[i], weight=0.5 * w_e)
            self.connect(nexts[i], recalls[(i - 1) % N], weight=0.5 * w_i)

            self.connect(prevs[i], recalls[i], weight=0.5 * w_e)
            self.connect(prevs[i], recalls[(i + 1) % N], weight=0.5 * w_i)


class RaceRouter(STICK):
    def __init__(self):
        super(RaceRouter, self).__init__()

        input0, input1 = self.create_neuron(['input0', 'input1'])
        reset = self.create_neuron('reset')
        init0, init1 = self.create_neuron(['init0', 'init1'])
        gate0, gate1 = self.create_neuron(['gate0', 'gate1'])
        tiebreak = self.create_neuron('tiebreak')
        output = self.create_neuron('output')

        self.set_input('input0', input0)
        self.set_input('input1', input1)
        self.set_input('reset', reset)

        self.set_output('output', output)

        self.connect(tiebreak, tiebreak)
        self.connect(tiebreak, init0)
        self.connect(tiebreak, init0, weight=w_i, delay=T_syn + T_neu)

        self.connect(input0, tiebreak, weight=0.5 * w_e)
        self.connect(input0, tiebreak, weight=0.5 * w_i, delay=T_syn + T_neu)
        self.connect(input0, init0, weight=w_e)
        self.connect(input0, init0, weight=w_i, delay=T_syn + T_neu)
        self.connect(input0, init1, weight=w_i)
        self.connect(input0, init1, weight=w_e, delay=2 * T_syn + T_neu)
        self.connect(input0, gate0, weight=0.5 * w_e, delay=3 * T_syn + 2 * T_neu)
        self.connect(input0, gate0, weight=0.5 * w_i, delay=4 * T_syn + 3 * T_neu)

        self.connect(input1, tiebreak, weight=0.5 * w_e)
        self.connect(input1, tiebreak, weight=0.5 * w_i, delay=T_syn + T_neu)
        self.connect(input1, init1, weight=w_e)
        self.connect(input1, init1, weight=w_i, delay=T_syn + T_neu)
        self.connect(input1, init0, weight=w_i)
        self.connect(input1, init0, weight=w_e, delay=2 * T_syn + T_neu)
        self.connect(input1, gate1, weight=0.5 * w_e, delay=3 * T_syn + 2 * T_neu)
        self.connect(input1, gate1, weight=0.5 * w_i, delay=4 * T_syn + 3 * T_neu)

        self.connect(init0, init1, weight=w_i)
        self.connect(init0, gate0, weight=0.5 * w_e)

        self.connect(init1, init0, weight=w_i)
        self.connect(init1, gate1, weight=0.5 * w_e)

        self.connect(gate0, gate0)
        self.connect(gate0, output)

        self.connect(gate1, gate1)
        self.connect(gate1, output)

        self.connect(reset, init0, weight=2 * w_e)
        self.connect(reset, init1, weight=2 * w_e)
        self.connect(reset, init0, weight=w_e, delay=2 * T_syn + T_neu)
        self.connect(reset, init1, weight=w_e, delay=2 * T_syn + T_neu)
        self.connect(reset, gate0, delay=2 * T_syn + T_neu)
        self.connect(reset, gate1, delay=2 * T_syn + T_neu)
        self.connect(reset, gate0, weight=w_i, delay=3 * T_syn + 2 * T_neu)
        self.connect(reset, gate1, weight=w_i, delay=3 * T_syn + 2 * T_neu)
        self.connect(reset, output, weight=2 * w_i, delay=3 * T_syn + 2 * T_neu)
