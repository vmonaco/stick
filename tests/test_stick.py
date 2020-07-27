from stick import *


def test_timesteps():
    # Decoded values for spikes within dt of each other should be equal
    assert isclose(decode(array([0, 20]) * ms), decode(array([0, (20 * ms + dt) / ms]) * ms), atol=abstol)
    assert isclose(decode(array([0, 20]) * ms), decode(array([0, (20 * ms - dt) / ms]) * ms), atol=abstol)

    # Timestamps will always be some multiple of dt. Anything greater than 1 dt is not equal
    assert not isclose(decode(array([0, 20]) * ms), decode(array([0, (20 * ms + 2 * dt) / ms]) * ms), atol=abstol)
    assert not isclose(decode(array([0, 20]) * ms), decode(array([0, (20 * ms - 2 * dt) / ms]) * ms), atol=abstol)

    print('Num unique values that can be encoded:', T_cod / T_neu)

    # Multiple connections send spikes separated by T_neu = 2*dt
    class FirstLast(STICK):
        def __init__(self):
            super(FirstLast, self).__init__()

            first, last = self.create_neuron(['first', 'last'])
            self.set_input('first', first)
            self.set_output('last', last)

            self.connect(first, last, weight=w_e, delay=T_syn)
            self.connect(first, last, weight=w_e, delay=T_syn + T_neu)
            self.connect(first, last, weight=w_e, delay=T_syn + 2 * T_neu)
            self.connect(first, last, weight=w_e, delay=T_syn + 3 * T_neu)

    m = FirstLast()

    spikes_out, statemon = m.run(5 * ms, spikes_in={
        'first': [0] * ms,
    }, return_statemon=True)

    # Should have one spike for each synapse
    assert len(spikes_out['last']) == 4


def test_nested(value1=0.3, value2=0.7, value3=0.2, delay_ms=8 * ms):
    # test STICK inside a STICK
    class Delay(STICK):
        def __init__(self, delay):
            super(Delay, self).__init__()

            node0, node1 = self.create_neuron(['node0', 'node1'])

            # Set the input
            self.set_input('input', node0)

            # Set the output
            self.set_output('output', node1)

            # Delayed connection between node1/node2
            self.connect(node0, node1, delay=delay)

    class TwoConstants(STICK):
        def __init__(self, value1, value2, delay_ms):
            super(TwoConstants, self).__init__()

            recall = self.create_neuron('recall')
            const1 = self.add_stick('const1', Constant(value1))
            const2 = self.add_stick('const2', Constant(value2))
            delay = self.add_stick('delay', Delay(delay_ms))

            # Set the input
            self.set_input('recall', recall)

            # Set the output
            self.set_output('output1', const1['output'])
            self.set_output('output2', const2['output'])

            # Recall the first constant immediately and the second after a delay
            self.connect(recall, const1['recall'])
            self.connect(recall, delay['input'])
            self.connect(delay['output'], const2['recall'])

    constant = TwoConstants(value1, value2, delay_ms)
    spikes_out = constant.run(200 * ms, spikes_in={'recall': [10 * ms]})
    assert isclose(decode(spikes_out['output1']), value1, atol=abstol)
    assert isclose(decode(spikes_out['output2']), value2, atol=abstol)

    # STICK inside a STICK inside a STICK
    class ThreeConstants(STICK):
        def __init__(self, value1, value2, value3, delay_ms):
            super(ThreeConstants, self).__init__()

            tc = self.add_stick('tc', TwoConstants(value1, value2, delay_ms))
            const3 = self.add_stick('const3', Constant(value3))
            recall = self.create_neuron('recall')

            self.set_input('recall', recall)
            self.set_output('out1', tc['output1'])
            self.set_output('out2', tc['output2'])
            self.set_output('out3', const3['output'])

            self.connect(recall, tc['recall'])
            self.connect(recall, const3['recall'])

    constant = ThreeConstants(value1, value2, value3, delay_ms)
    spikes_out = constant.run(200 * ms, spikes_in={'recall': [10 * ms]})

    assert isclose(decode(spikes_out['out1']), value1, atol=abstol)
    assert isclose(decode(spikes_out['out2']), value2, atol=abstol)
    assert isclose(decode(spikes_out['out3']), value3, atol=abstol)


def test_encoding():
    x = [0.3, 0.7, 0.2]

    t_sequential = encode(x, method='sequential')
    t_chained = encode(x, method='chained')
    t_superimposed = encode(x, method='superimposed')

    assert (len(t_sequential) == 2 * len(x))
    assert (len(t_chained) == len(x) + 1)
    assert (len(t_superimposed) == len(x) + 1)

    x_sequential = decode(t_sequential, method='sequential')
    x_chained = decode(t_chained, method='chained')
    x_superimposed = decode(t_superimposed, method='superimposed')

    assert isclose(x, x_sequential, atol=abstol).all()
    assert isclose(sort(x), x_chained, atol=abstol).all()
    assert isclose(sort(x), x_superimposed, atol=abstol).all()
