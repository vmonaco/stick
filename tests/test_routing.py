from stick import *


def test_gate(values=[0.2, 0.4, 0.3], out=None):
    m = Gate()

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'input': [50, 80, 125, 150, 210, 250] * ms,
        'on': [110] * ms,
        'off': [180] * ms
    },
                                 return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['input', 'on', 'off', 'gate', 'output'],
                    show_values=['input', 'output'])

    assert len(spikes_out['output']) == 2

    input_spikes = encode(values, t_0=10 * ms, spacing=20 * ms)
    spikes_out, statemon = m.run(500 * ms, spikes_in={
        'input': input_spikes,
        'on': [input_spikes[1] + 10 * ms],
        'off': [input_spikes[3] + 10 * ms]
    },
                                 return_statemon=True)

    assert len(spikes_out['output']) == 2

    values_decoded = decode(spikes_out['output'])

    assert isclose(values_decoded[0], values[1], atol=abstol)


def test_toggle(values=[0.2, 0.4, 0.3], out=None):
    m = Toggle()

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'input': [50, 80, 125, 150, 160, 210, 250] * ms,
        'switch': [110, 230] * ms},
                                 return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['input', 'switch', 'first', 'last', 'init', 'out0', 'out1'])

    assert len(spikes_out['out0']) == 3
    assert len(spikes_out['out1']) == 4

    input_spikes = encode(values, t_0=10 * ms, spacing=20 * ms)
    spikes_out, statemon = m.run(500 * ms, spikes_in={
        'input': input_spikes,
        'switch': [input_spikes[1] + 10 * ms, input_spikes[3] + 10 * ms]},
                                 return_statemon=True)

    assert len(spikes_out['out0']) == 4
    assert len(spikes_out['out1']) == 2

    output1_values = decode(spikes_out['out0'])
    output2_values = decode(spikes_out['out1'])

    assert isclose(output1_values[0], values[0], atol=abstol)
    assert isclose(output1_values[1], values[2], atol=abstol)
    assert isclose(output2_values[0], values[1], atol=abstol)


def test_linked_router(delay=25 * ms, out=None):
    m = LinkedRouter(N=3)

    values = [0.3, 0.1, 0.2]
    N = len(values)

    next_spikes, input_spikes = [], []
    next_t_0 = delay
    for value in values:
        next_spikes.append(next_t_0)
        next_t_0 = next_spikes[-1] + delay
        input_spikes.extend(encode(value, t_0=next_t_0))
        next_t_0 = input_spikes[-1] + delay

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'next': next_spikes,
        'input': input_spikes,
    }, return_statemon=True)

    plot_chronogram(statemon, out=out, subset=['next', 'input', 'out0', 'out1', 'out2'],
                    show_values=['input', 'out0', 'out1', 'out2'])

    assert len(spikes_out['out0']) == 2
    assert len(spikes_out['out1']) == 2
    assert len(spikes_out['out2']) == 2


def test_race_router(out=None):
    m = RaceRouter()

    spikes_in = {
        'input0': [10, 75, 170, 240] * ms,
        'input1': [40, 90, 150, 190] * ms,
        'reset': [120] * ms
    }

    spikes_out, statemon = m.run(300 * ms, spikes_in=spikes_in,
                                 return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['input0', 'input1', 'reset', 'init0', 'init1', 'gate0', 'gate1', 'output'],
                    show_values=['input0', 'input1', 'output'])
    print(spikes_out['output'])

    assert len(spikes_out['output']) == len(spikes_in['input0'])

    assert isclose(decode(spikes_in['input0'][:2], method='chained'),
                   decode(spikes_out['output'][:2], method='chained'), atol=abstol).all()

    assert isclose(decode(spikes_in['input1'][2:], method='chained'),
                   decode(spikes_out['output'][2:], method='chained'), atol=abstol).all()
