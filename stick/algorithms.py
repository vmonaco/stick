from .stick import *
from .memory import *


class Sorting_v1(STICK):
    def __init__(self, N):
        super(Sorting_v1, self).__init__()

        assert N >= 2

        store_, sort_, recall = self.create_neuron(['input', 'sort', 'recall'])
        output, store_ready, sort_ready, recall_ready = self.create_neuron(
            ['output', 'store_ready', 'sort_ready', 'recall_ready'])
        first = self.create_neuron('first')
        ll_input = self.add_stick('input_list', LinkedList(N))
        ll_sorted = self.add_stick('sorted_list', LinkedList(N))

        self.set_input('input', store_)
        self.set_input('sort', sort_)
        self.set_input('recall', recall)

        self.set_output('output', output)
        self.set_output('store_ready', store_ready)
        self.set_output('sort_ready', sort_ready)
        self.set_output('recall_ready', recall_ready)

        self.connect(store_, first)
        self.connect(store_, ll_input['store'], delay=2 * T_syn + T_neu)
        self.connect(store_, store_ready, weight=w_e / (2 * N))
        self.connect(first, first, weight=w_i)
        self.connect(first, ll_input['next'])

        self.connect(sort_, ll_input['p_recall'])
        self.connect(sort_, ll_sorted['store'], weight=w_i)
        self.connect(sort_, ll_sorted['next'], weight=w_i)
        self.connect(sort_, ll_sorted['p_store'], delay=6 * T_syn + 4 * T_neu)

        self.connect(ll_input['output'], ll_sorted['next'])
        self.connect(ll_input['output'], ll_sorted['store'])
        self.connect(ll_input['output'], sort_ready, weight=w_e / (N + 1))

        self.connect(recall, ll_sorted['next'])
        self.connect(recall, ll_sorted['recall'])

        self.connect(output, ll_sorted['next'], weight=0.5 * w_e)
        self.connect(output, ll_sorted['recall'], weight=0.5 * w_e)

        self.connect(ll_sorted['output'], output)

        self.connect(ll_sorted['output'], recall_ready, weight=w_e / (2 * N))
        self.connect(recall_ready, ll_sorted['next'], weight=w_i)
        self.connect(recall_ready, ll_sorted['recall'], weight=w_i)


class Sorting(STICK):
    def __init__(self, N):
        super(Sorting, self).__init__()

        assert N >= 2

        sort_, ready = self.create_neuron(['sort', 'ready'])
        memory_in = self.add_stick('memory_in', Superimposed(N))
        memory_out = self.add_stick('memory_out', Sequential(N, spacing=5 * T_syn))
        router = self.add_stick('router', LinkedRouter(N + 1))

        self.set_input('sort', sort_)
        self.set_input('recall', memory_out['recall'])
        self.set_output('output', memory_out['output'])
        self.set_output('ready', ready)

        self.connect(sort_, memory_in['recall'])
        self.connect(memory_in['output'], router['next'])
        self.connect(memory_in['output'], router['input'])

        for i in range(N):
            self.connect(router['out0'], memory_out['store%d' % i])

        self.connect(router['out%d' % N], ready)

        for i in range(N):
            self.connect(router['out%d' % (i + 1)], memory_out['store%d' % i])
            self.set_input('store%d' % i, memory_in['store%d' % i])


class Searching_v1(STICK):
    def __init__(self, N):
        super(Searching_v1, self).__init__()

        assert N >= 2

        store_, search = self.create_neuron(['input', 'search'])
        output = self.create_neuron('output')
        store_first = self.create_neuron('store_first')
        reset = self.create_neuron('reset')
        first, last = self.create_neuron(['first', 'last'])
        acc_min, acc_max, acc = self.create_neuron(['acc_min', 'acc_max', 'acc'])
        llist = self.add_stick('list', LinkedList(N))
        tog0 = self.add_stick('tog0', Toggle2())
        tog1 = self.add_stick('tog1', Toggle2())
        tog2 = self.add_stick('tog2', Toggle2())

        self.set_input('input', store_)
        self.set_input('search', search)

        self.set_output('output', output)

        self.connect(store_, store_first)
        self.connect(store_, llist['store'], delay=2 * T_syn + T_neu)

        self.connect(store_first, store_first, weight=w_i)
        self.connect(store_first, llist['next'])

        self.connect(llist['output'], tog0['input'])

        self.connect(tog0['out0'], tog1['input'])

        self.connect(tog1['out0'], acc_min)
        self.connect(tog1['out0'], acc_min, delay=T_syn + T_neu)
        self.connect(tog1['out0'], acc_min, 'g_e', weight=-w_acc, delay=T_syn + T_neu)

        self.connect(tog1['out1'], output)
        self.connect(tog1['out1'], acc, 'g_e', weight=-w_acc)
        self.connect(tog1['out1'], acc_min, 'g_e', weight=-w_acc)
        self.connect(tog1['out1'], acc_max, 'g_e', weight=2 * w_acc)
        self.connect(tog1['out1'], tog0['switch'])

        self.connect(search, first)
        self.connect(search, last, weight=0.5 * w_e)

        delay = 11 * T_syn + 10 * T_neu
        self.connect(first, first, weight=w_i)
        self.connect(first, llist['p_recall'])
        self.connect(first, acc, delay=T_syn + delay)
        self.connect(first, acc, delay=T_syn + T_neu + delay)
        self.connect(first, acc, 'g_e', weight=-w_acc, delay=T_syn + T_neu + delay)
        self.connect(first, acc_max, delay=T_syn + delay)
        self.connect(first, acc_max, delay=T_syn + T_neu + delay)
        self.connect(first, acc_max, 'g_e', weight=-w_acc, delay=T_syn + T_neu + delay)

        self.connect(last, tog1['switch'], delay=T_syn + delay)
        self.connect(last, tog2['switch'], delay=T_syn + delay)
        self.connect(last, acc, 'g_e', weight=2 * w_acc, delay=T_syn + delay)
        self.connect(last, acc_min, 'g_e', weight=2 * w_acc, delay=T_syn + delay)

        self.connect(acc, output)

        self.connect(acc_min, tog2['input'])

        self.connect(tog2['out1'], output)
        self.connect(tog2['out1'], acc_max, 'g_e', weight=w_acc)

        self.connect(acc_max, output)

        self.connect(output, reset, weight=0.5 * w_e)
        self.connect(reset, tog0['switch'])
        self.connect(reset, tog1['switch'])
        self.connect(reset, tog2['switch'])


class Searching(STICK):
    def __init__(self, N):
        super(Searching, self).__init__()

        assert N >= 2

        search, output = self.create_neuron(['search', 'output'])
        memory = self.add_stick('memory', Superimposed(N))
        router = self.add_stick('router', LinkedRouter(4))
        race = self.add_stick('race', RaceRouter())

        first, last = self.create_neuron(['first', 'last'])
        acc, acc_max, acc_diff = self.create_neuron(['acc', 'acc_max', 'acc_diff'])

        self.set_input('search', search)
        self.set_output('output', race['output'])

        for i in range(N):
            self.set_input('store%d' % i, memory['store%d' % i])

        self.connect(search, first)
        self.connect(search, last, weight=0.5 * w_e)

        self.connect(memory['output'], router['input'])

        self.connect(first, first, weight=w_i)
        self.connect(first, memory['recall'])
        self.connect(first, router['next'], delay=4 * T_syn + 2 * T_neu)

        self.connect(last, router['next'], delay=4 * T_syn + 2 * T_neu)
        self.connect(last, acc, 'g_e', weight=2 * w_acc, delay=9 * T_syn + 8 * T_neu)
        self.connect(last, acc_diff, 'g_e', weight=2 * w_acc, delay=9 * T_syn + 8 * T_neu)

        self.connect(router['out0'], router['next'])
        self.connect(router['out0'], acc)
        self.connect(router['out0'], acc, 'g_e', weight=-w_acc)
        self.connect(router['out0'], acc_max)
        self.connect(router['out0'], acc_max, 'g_e', weight=-w_acc)

        self.connect(router['out1'], acc_diff)
        self.connect(router['out1'], acc_diff, delay=T_syn + T_neu)
        self.connect(router['out1'], acc_diff, 'g_e', weight=-w_acc, delay=T_syn + T_neu)
        self.connect(router['out1'], race['input0'], weight=w_i, delay=2 * T_syn + T_neu)

        self.connect(router['out2'], router['next'], delay=3 * T_syn)
        self.connect(router['out2'], race['input1'], delay=2 * T_syn + 2 * T_neu)
        self.connect(router['out2'], acc_max, 'g_e', weight=2 * w_acc, delay=T_syn + T_neu)

        self.connect(acc, race['input0'])
        self.connect(acc_max, race['input1'])
        self.connect(acc_diff, race['input0'], delay=T_syn + T_neu)
