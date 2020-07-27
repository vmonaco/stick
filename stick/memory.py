from .stick import *
from .routing import *
from .primitives import *
from .util import encode


### Memory primitives

class Constant(STICK):
    def __init__(self, value):
        super(Constant, self).__init__()

        recall, output = self.create_neuron(['recall', 'output'])

        # Set the inputs and outputs
        self.set_input('recall', recall)
        self.set_output('output', output)

        # Form the internal connections
        self.connect(recall, output)
        self.connect(recall, output, delay=T_syn + encode(value)[1])


class Boolean(STICK):
    def __init__(self):
        super(Boolean, self).__init__()

        set_, recall, reset, boolean, out = self.create_neuron(['set', 'recall', 'reset', 'bool', 'output'])
        self.set_input('set', set_)
        self.set_input('recall', recall)
        self.set_input('reset', reset)
        self.set_output('output', out)

        self.connect(set_, boolean, weight=w_e)
        self.connect(set_, boolean, weight=-0.5 * w_e, delay=T_syn + T_neu)
        self.connect(set_, out, weight=w_i)  # , delay=2 * T_syn + T_neu)

        self.connect(reset, boolean, weight=w_e)
        self.connect(reset, boolean, weight=w_i, delay=T_syn + T_neu)
        self.connect(reset, out, weight=w_i)  # , delay=2 * T_syn + T_neu)

        self.connect(recall, boolean, weight=0.5 * w_e)
        self.connect(recall, boolean, weight=-0.5 * w_e, delay=T_syn + T_neu)

        self.connect(boolean, boolean, weight=w_e)
        self.connect(boolean, out, weight=w_e)


class Timer(STICK):
    def __init__(self, extra_delay=0 * ms):
        super(Timer, self).__init__()

        store_, recall = self.create_neuron(['store', 'recall'])
        first, last = self.create_neuron(['first', 'last'])
        acc, output = self.create_neuron(['acc', 'output'])
        # ready = self.create_neuron('ready')

        self.set_input('store', store_)
        self.set_input('recall', recall)
        self.set_output('output', output)
        # self.set_output('ready', ready)

        self.connect(store_, first)
        self.connect(store_, last, weight=0.5 * w_e)

        self.connect(first, first, weight=w_i)

        self.connect(first, acc, delay=3 * T_syn + 2 * T_neu + extra_delay)
        self.connect(first, acc, 'g_e', weight=-w_acc, delay=3 * T_syn + 2 * T_neu + extra_delay)

        self.connect(last, acc, 'g_e', weight=w_acc)
        # self.connect(last, ready)

        self.connect(acc, output)

        self.connect(recall, acc, 'g_e', weight=w_acc)


class Fragile(STICK):
    def __init__(self, extra_delay=0 * ms):
        super(Fragile, self).__init__()

        store_, recall = self.create_neuron(['store', 'recall'])
        first, last = self.create_neuron(['first', 'last'])
        acc, output = self.create_neuron(['acc', 'output'])

        self.set_input('store', store_)
        self.set_input('recall', recall)
        self.set_output('output', output)

        self.connect(store_, first)
        self.connect(store_, last, weight=0.5 * w_e)

        self.connect(first, first, weight=w_i)

        self.connect(first, acc, delay=2 * T_syn)
        self.connect(first, acc, 'g_e', weight=-w_acc, delay=2 * T_syn)

        self.connect(last, acc, 'g_e', weight=w_acc)

        self.connect(acc, output)

        self.connect(recall, output)
        self.connect(recall, acc, 'g_e', weight=w_acc)


class Volatile(STICK):
    def __init__(self):
        super(Volatile, self).__init__()

        store_, recall = self.create_neuron(['store', 'recall'])
        first, last = self.create_neuron(['first', 'last'])
        acc, output = self.create_neuron(['acc', 'output'])
        ready = self.create_neuron('ready')

        self.set_input('store', store_)
        self.set_input('recall', recall)
        self.set_output('output', output)
        self.set_output('ready', ready)

        self.connect(store_, first)
        self.connect(store_, last, weight=0.5 * w_e)

        self.connect(first, first, weight=w_i)
        self.connect(first, acc)
        self.connect(first, output, weight=w_i)

        self.connect(first, acc, delay=T_syn + T_neu)
        self.connect(first, acc, 'g_e', weight=-w_acc, delay=T_syn + T_neu)

        self.connect(last, acc, 'g_e', weight=w_acc, delay=T_syn + T_neu)
        self.connect(last, ready, delay=T_syn + T_neu)

        self.connect(acc, output)

        self.connect(recall, acc, 'g_e', weight=w_acc)
        self.connect(recall, output, delay=2 * T_syn)


class Persistent(STICK):
    def __init__(self):
        super(Persistent, self).__init__()

        store_, recall = self.create_neuron(['store', 'recall'])
        last, output = self.create_neuron(['last', 'output'])
        ready = self.create_neuron('ready')

        vmem0 = self.add_stick('vmem0', Volatile())
        vmem1 = self.add_stick('vmem1', Volatile())
        toggle0 = self.add_stick('tog0', Toggle())
        toggle1 = self.add_stick('tog1', Toggle())

        self.set_input('store', store_)
        self.set_input('recall', recall)
        self.set_output('output', output)
        self.set_output('ready', ready)

        self.connect(store_, toggle0['input'])
        self.connect(recall, toggle1['input'])

        self.connect(toggle0['out0'], vmem0['store'])
        self.connect(toggle0['out1'], vmem1['store'])

        self.connect(toggle1['out0'], vmem0['recall'])
        self.connect(toggle1['out1'], vmem1['recall'])

        self.connect(vmem0['output'], output)
        self.connect(vmem0['output'], vmem1['store'])

        self.connect(vmem1['output'], output)
        self.connect(vmem1['output'], vmem0['store'])

        self.connect(vmem0['ready'], ready)
        self.connect(vmem1['ready'], ready)

        self.connect(output, last, weight=0.5 * w_e)
        self.connect(last, toggle0['switch'])
        self.connect(last, toggle1['switch'])


class Parallel(STICK):
    def __init__(self):
        super(Parallel, self).__init__()

        restore, recall = self.create_neuron(['restore', 'recall'])

        first, last, output = self.create_neuron(['first', 'last', 'output'])
        gate0, gate1 = self.create_neuron(['gate0', 'gate1'])
        ready = self.create_neuron('ready')

        vmem0 = self.add_stick('vmem0', Volatile())
        vmem1 = self.add_stick('vmem1', Volatile())
        toggle0 = self.add_stick('tog0', Toggle())
        toggle1 = self.add_stick('tog1', Toggle())

        self.set_input('restore', restore)
        self.set_input('recall', recall)
        self.set_output('output', output)
        self.set_output('ready', ready)

        self.connect(restore, toggle0['input'])
        self.connect(restore, first)

        self.connect(first, first, weight=w_i)
        self.connect(first, toggle1['input'])

        self.connect(recall, toggle1['input'])

        self.connect(toggle0['out0'], vmem1['store'])
        self.connect(toggle0['out0'], gate0, weight=w_i)

        self.connect(toggle0['out1'], vmem0['store'])
        self.connect(toggle0['out1'], gate1, weight=w_i)

        self.connect(toggle1['out0'], vmem0['recall'])
        self.connect(toggle1['out1'], vmem1['recall'])

        self.connect(vmem0['output'], output)
        self.connect(vmem0['output'], gate0)
        self.connect(gate0, vmem1['store'])

        self.connect(vmem1['output'], output)
        self.connect(vmem1['output'], gate1)
        self.connect(gate1, vmem0['store'])

        self.connect(vmem0['ready'], ready)
        self.connect(vmem1['ready'], ready)

        # TODO: fix delay here
        self.connect(output, last, weight=0.5 * w_e, delay=10 * T_syn)
        self.connect(last, toggle0['switch'])
        self.connect(last, toggle1['switch'])


### Memory banks

class Sequential(STICK):
    def __init__(self, N, spacing=5 * T_syn):
        super(Sequential, self).__init__()

        recall, output = self.create_neuron(['recall', 'output'])
        self.set_input('recall', recall)
        self.set_output('output', output)
        self.connect(recall, output, delay=2 * T_syn)

        for i in range(N):
            vmem = self.add_stick('vmem%d' % i, Timer())
            self.set_input('store%d' % i, vmem['store'])
            self.connect(vmem['output'], output)

            if i < N - 1:
                self.connect(vmem['output'], output, delay=spacing + T_syn)

            if i == 0:
                self.connect(recall, vmem['recall'])
            else:
                self.connect(prev_vmem['output'], vmem['recall'], delay=spacing)

            prev_vmem = vmem


class Chained(STICK):
    def __init__(self, N):
        super(Chained, self).__init__()

        recall, output = self.create_neuron(['recall', 'output'])
        self.set_input('recall', recall)
        self.set_output('output', output)
        self.connect(recall, output)

        for i in range(N):
            vmem = self.add_stick('vmem%d' % i, Timer(extra_delay=T_syn))
            self.set_input('store%d' % i, vmem['store'])
            self.connect(vmem['output'], output)

            if i == 0:
                self.connect(recall, vmem['recall'])
            else:
                self.connect(prev_vmem['output'], vmem['recall'])

            prev_vmem = vmem


class Superimposed(STICK):
    def __init__(self, N):
        super(Superimposed, self).__init__()

        recall, output = self.create_neuron(['recall', 'output'])
        self.set_input('recall', recall)
        self.set_output('output', output)
        self.connect(recall, output, delay=2 * T_syn)

        for i in range(N):
            vmem = self.add_stick('vmem%d' % i, Timer())
            self.set_input('store%d' % i, vmem['store'])
            self.connect(vmem['output'], output)
            self.connect(recall, vmem['recall'])


class SpatialMemory(STICK):
    def __init__(self, N):
        super(SpatialMemory, self).__init__()

        assert N >= 2

        recall, reset = self.create_neuron(['recall', 'reset'])

        self.set_input('recall', recall)
        self.set_input('reset', reset)

        for i in range(N):
            input_, output = self.create_neuron(['set%d' % i, 'out%d' % i])
            boolean = self.add_stick('bool%d' % i, Boolean())
            self.set_input('set%d' % i, input_)
            self.set_output('out%d' % i, output)
            self.connect(input_, boolean['set'], weight=w_e)
            self.connect(boolean['output'], output, weight=w_e)
            self.connect(recall, boolean['recall'], weight=w_e)
            self.connect(reset, boolean['reset'], weight=w_e)


### Data structures

class AddressableMemory(STICK):
    def __init__(self, N):
        super(AddressableMemory, self).__init__()

        assert N >= 2

        addresser = self.add_stick('addresser', Addresser(N))
        store_mem = self.add_stick('storemem', SpatialMemory(N))
        recall_mem = self.add_stick('recallmem', SpatialMemory(N))

        store_, locate, recall = self.create_neuron(['store', 'locate', 'recall'])
        first = self.create_neuron('first')
        l_ready, s_ready, output = self.create_neuron(['l_ready', 's_ready', 'output'])

        self.set_input('store', store_)
        self.set_input('locate', locate)
        self.set_input('recall', recall)

        self.set_output('l_ready', l_ready)
        self.set_output('s_ready', s_ready)
        self.set_output('output', output)

        self.connect(locate, addresser['locate'], weight=w_e)
        self.connect(locate, first, weight=w_e)
        self.connect(locate, l_ready, weight=0.5 * w_e, delay=6 * T_syn + 5 * T_neu)

        self.connect(first, first, weight=w_i)
        self.connect(first, store_mem['reset'], weight=w_e)
        self.connect(first, recall_mem['reset'], weight=w_e)

        self.connect(store_, store_mem['recall'], weight=w_e)
        self.connect(recall, recall_mem['recall'], weight=w_e)

        for i in range(N):
            pmem = self.add_stick('pmem%d' % i, Persistent())

            self.connect(addresser['addr%d' % i], store_mem['set%d' % i], weight=w_e)
            self.connect(addresser['addr%d' % i], recall_mem['set%d' % i], weight=w_e)

            self.connect(store_mem['out%d' % i], pmem['store'], weight=w_e)
            self.connect(recall_mem['out%d' % i], pmem['recall'], weight=w_e)

            self.connect(pmem['output'], output, weight=w_e)
            self.connect(pmem['ready'], s_ready, weight=w_e)


class LinkedList(STICK):
    def __init__(self, N):
        super(LinkedList, self).__init__()

        # if next+store spike same time, behavior is next,store. next must spike before first store
        store_, recall, next_ = self.create_neuron(['store', 'recall', 'next'])
        p_store, p_recall = self.create_neuron(['p_store', 'p_recall'])
        l_ready, s_ready, output = self.create_neuron(['l_ready', 's_ready', 'output'])
        linked_addresser = self.add_stick('addresser', LinkedMemRouter(N))

        self.set_input('store', store_)
        self.set_input('recall', recall)
        self.set_input('next', next_)
        self.set_input('p_store', p_store)
        self.set_input('p_recall', p_recall)

        self.set_output('l_ready', l_ready)
        self.set_output('s_ready', s_ready)
        self.set_output('output', output)

        self.connect(store_, linked_addresser['store'], weight=w_e)
        self.connect(recall, linked_addresser['recall'], weight=w_e)
        self.connect(next_, linked_addresser['next'], weight=w_e)

        for i in range(N):
            # mem = self.add_stick('pmem%d' % i, Persistent())
            mem = self.add_stick('pmem%d' % i, Volatile())

            self.connect(linked_addresser['store%d' % i], mem['store'], weight=w_e)
            self.connect(linked_addresser['recall%d' % i], mem['recall'], weight=w_e)

            # Delayed to the same as if they were called in parallel through the addresser
            self.connect(p_store, mem['store'], weight=w_e, delay=6 * T_syn + 5 * T_neu)
            self.connect(p_recall, mem['recall'], weight=w_e)

            self.connect(mem['output'], output, weight=w_e)
            self.connect(mem['ready'], s_ready, weight=w_e)


class DoublyLinkedList(STICK):
    def __init__(self, N):
        super(DoublyLinkedList, self).__init__()

        store_, recall = self.create_neuron(['store', 'recall'])
        next_, prev = self.create_neuron(['next', 'prev'])
        p_store, p_recall = self.create_neuron(['p_store', 'p_recall'])
        l_ready, s_ready, output = self.create_neuron(['l_ready', 's_ready', 'output'])

        addresser = self.add_stick('addresser', DoublyLinkedMemRouter(N))

        self.set_input('store', store_)
        self.set_input('recall', recall)
        self.set_input('next', next_)
        self.set_input('prev', prev)
        self.set_input('p_store', p_store)
        self.set_input('p_recall', p_recall)

        self.set_output('l_ready', l_ready)
        self.set_output('s_ready', s_ready)
        self.set_output('output', output)

        self.connect(next_, addresser['next'], weight=w_e)
        self.connect(prev, addresser['prev'], weight=w_e)

        self.connect(store_, addresser['store'], weight=w_e)
        self.connect(recall, addresser['recall'], weight=w_e)

        for i in range(N):
            # pmem = self.add_stick('pmem%d' % i, Persistent())
            pmem = self.add_stick('pmem%d' % i, Volatile())

            self.connect(p_store, pmem['store'])
            self.connect(p_recall, pmem['recall'])

            self.connect(addresser['store%d' % i], pmem['store'])
            self.connect(addresser['recall%d' % i], pmem['recall'])

            self.connect(pmem['output'], output, weight=w_e)
            self.connect(pmem['ready'], s_ready, weight=w_e)
