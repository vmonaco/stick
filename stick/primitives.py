from .stick import *


# TODO: binary to interval converter?

class GeSynapse(STICK):
    def __init__(self, N):
        super(GeSynapse, self).__init__()

        pre, post = self.create_neuron(['pre', 'post'])
        aux = self.create_neuron(['aux%d' % i for i in range(N)])

        self.set_input('pre', pre)
        self.set_output('post', post)

        for i in range(N):
            self.connect(pre, aux[i], weight=w_e / (i+1))
            self.connect(aux[i], aux[i], weight=w_e)
            self.connect(aux[i], post, weight=V_t/(T_max/T_syn))
            self.connect(post, aux[i], weight=w_e)
            self.connect(post, aux[i], weight=w_i, delay=T_syn + T_neu)
            self.connect(post, post, weight=-N*V_t/(T_max/T_syn), delay=T_syn+T_neu)


class Iterator(STICK):
    def __init__(self, N):
        super(Iterator, self).__init__()

        next_, init = self.create_neuron(['next', 'init'])
        addr = self.create_neuron(['addr%d' % i for i in range(N)])

        self.set_input('next', next_)

        self.connect(next_, init)
        self.connect(next_, init, weight=w_i, delay=T_syn + T_neu)

        for i in range(N):
            self.set_output(addr[i].name, addr[i])
            self.connect(init, addr[i], weight=(N - i - 1) * w_e / N)
            self.connect(next_, addr[i], weight=w_e / N, delay=2 * T_syn + T_neu)


# TODO: next/prev currently take at least 2 time steps, possible with 1 step?
class DoublyIterator(STICK):
    def __init__(self, N):
        super(DoublyIterator, self).__init__()

        next_, prev_ = self.create_neuron(['next', 'prev'])
        n_init, p_init, inhibit = self.create_neuron(['n_init', 'p_init', 'inhibit'])
        nexts = self.create_neuron(['next%d' % i for i in range(N)])
        prevs = self.create_neuron(['prev%d' % i for i in range(N)])
        outs = self.create_neuron(['out%d' % i for i in range(N)])

        self.set_input('next', next_)
        self.set_input('prev', prev_)

        self.connect(next_, n_init)
        self.connect(next_, n_init, weight=w_i, delay=T_syn + T_neu)
        self.connect(next_, p_init, weight=w_i, delay=T_syn + T_neu)
        self.connect(prev_, p_init)
        self.connect(prev_, p_init, weight=w_i, delay=T_syn + T_neu)
        self.connect(prev_, n_init, weight=w_i, delay=T_syn + T_neu)

        self.connect(n_init, nexts[0], weight=0.5 * w_e)
        self.connect(n_init, prevs[-2 % N], weight=0.5 * w_e)

        self.connect(p_init, prevs[-1 % N], weight=0.5 * w_e)
        self.connect(p_init, nexts[1 % N], weight=0.5 * w_e)

        for i in range(N):
            self.set_output(outs[i].name, outs[i])

            # Inputs send +0.5 quickly followed by -0.5. Only "on" neurons will spike
            self.connect(next_, nexts[i], weight=0.5 * w_e, delay=2 * T_syn + T_neu)
            self.connect(next_, nexts[i], weight=-0.5 * w_e, delay=2 * T_syn + 2 * T_neu)
            self.connect(prev_, prevs[i], weight=0.5 * w_e, delay=2 * T_syn + T_neu)
            self.connect(prev_, prevs[i], weight=-0.5 * w_e, delay=2 * T_syn + 2 * T_neu)

            # Next: chain forward, prev: chain backward
            self.connect(nexts[i], nexts[(i + 1) % N], weight=0.5 * w_e, delay=T_neu)
            self.connect(prevs[i], prevs[(i - 1) % N], weight=0.5 * w_e, delay=T_neu)

            # Recurrent connections to reset after spiking (counteracts the 2nd inhibitory synapse from input)
            self.connect(nexts[i], nexts[i], weight=0.5 * w_e, delay=T_neu)
            self.connect(prevs[i], prevs[i], weight=0.5 * w_e, delay=T_neu)

            # Next: enable i-1 prev and disable i-2 prev
            self.connect(nexts[i], prevs[(i - 1) % N], weight=0.5 * w_e, delay=T_neu)
            self.connect(nexts[i], prevs[(i - 2) % N], weight=-0.5 * w_e, delay=T_neu)

            # Prev: enable current next and disable last next
            self.connect(prevs[i], nexts[(i + 1) % N], weight=0.5 * w_e, delay=T_neu)
            self.connect(prevs[i], nexts[(i + 2) % N], weight=-0.5 * w_e, delay=T_neu)

            # Outputs
            self.connect(nexts[i], outs[i])
            self.connect(prevs[i], outs[i])


# TODO: verify this works
class BinaryIterator(STICK):
    def __init__(self, N):
        super(BinaryIterator, self).__init__()

        assert 2 ** int(log2(N)) == N  # Only works with powers of 2

        num_levels = int(log2(N))

        next_ = self.create_neuron('next')
        self.set_input('next', next_)

        def create_tree(parent, path='', level=0):
            left = '0' + path
            right = '1' + path

            if level < num_levels - 1:
                first, last = self.create_neuron(['first' + left, 'last' + right])
            else:
                first_name = 'out%d' % int(left, base=2)
                last_name = 'out%d' % int(right, base=2)
                first, last = self.create_neuron([first_name, last_name])
                self.set_output(first_name, first)
                self.set_output(last_name, last)

            self.connect(parent, first)
            self.connect(parent, last, weight=0.5 * w_e)
            self.connect(first, first, weight=w_i)

            if level < num_levels - 1:
                create_tree(first, path=left, level=level + 1)
                create_tree(last, path=right, level=level + 1)

        create_tree(next_)


class Addresser(STICK):
    def __init__(self, N):
        super(Addresser, self).__init__()

        assert N >= 2

        w_acc_n = N * (V_t * tau_m / (T_cod - ((N - 1) * (T_syn + T_neu))))

        input_ = self.create_neuron('locate')
        first, last = self.create_neuron(['first', 'last'])
        acc = self.create_neuron(['acc%d' % i for i in range(N)])
        addr = self.create_neuron(['addr%d' % i for i in range(N)])

        self.set_input('locate', input_)
        for i in range(N):
            self.set_output('addr%d' % i, addr[i])
            self.set_output('acc%d' % i, acc[i])

        self.connect(input_, first)
        self.connect(input_, last, weight=0.5 * w_e)

        self.connect(first, first, weight=w_i)
        self.connect(first, acc[0], 'g_e', weight=w_acc_n, delay=T_min + T_syn)
        self.connect(first, addr[0], weight=0.5 * w_e)

        for i in range(N):
            self.connect(last, addr[i], weight=0.5 * w_e)
            self.connect(last, addr[i], weight=-0.5 * w_e, delay=T_syn + T_neu)

            self.connect(acc[i], addr[i], weight=-0.5 * w_e)

            self.connect(addr[i], acc[i], weight=w_e)
            self.connect(addr[i], addr[i], weight=w_e)

            if i < N - 1:
                self.connect(acc[i], addr[i + 1], weight=0.5 * w_e)
                self.connect(acc[i], acc[i + 1], 'g_e', weight=w_acc_n)
                self.connect(addr[i], addr[i + 1], weight=-0.5 * w_e)
                self.connect(addr[i], acc[i + 1], 'g_e', weight=-w_acc_n, delay=2 * T_syn + T_neu)
