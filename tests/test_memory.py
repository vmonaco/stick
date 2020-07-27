from stick import *


def test_constant(const=0.3):
    constant = Constant(const)
    spikes_out, statemon = constant.run(300 * ms, spikes_in={'recall': [10, 150] * ms}, return_statemon=True)

    values_out = decode(spikes_out['output'])
    assert isclose(values_out, const, atol=abstol).all()


def test_boolean(out=None):
    m = Boolean()

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'set': [10, 140] * ms,
        'recall': [60, 120, 200, 250] * ms,
        'reset': [90] * ms,
    }, return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['set', 'recall', 'reset', 'bool', 'output'])

    assert len(spikes_out['output']) == 3


def test_timer(value=0.3, out=None):
    m = Timer()

    recall_spikes = [120] * ms
    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'store': encode(value, t_0=10 * ms),
        'recall': recall_spikes
    }, return_statemon=True)

    plot_chronogram(statemon, out=out, subset=['store', 'recall', 'first', 'last', 'acc', 'output'],
                    show_values=['store'])

    assert len(spikes_out['output']) == 1
    assert isclose(value, decode(merge(recall_spikes, spikes_out['output'])), atol=abstol)


def test_volatile(value1=1., value2=0., value3=0.3, out=None):
    m = Volatile()

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'store': encode(value3, t_0=10 * ms),
        'recall': [120] * ms
    }, return_statemon=True)
    plot_chronogram(statemon, out=out,
                    subset=['store', 'recall', 'acc', 'first', 'last', 'ready', 'output'])

    plot_spikes(statemon, out='volatile-memory-spikes.pdf',
                subset=['store', 'recall', 'acc', 'first', 'last', 'ready', 'output'])

    # 0,        120,      240,      360,      480,      600       720         840
    # encode 1, recall 1, encode 1, encode 0, recall 0, recall ?, encode 0.3, recall 0.3

    store_spikes = merge(encode(value1),
                         encode(value1, t_0=240 * ms),
                         encode(value2, t_0=360 * ms),
                         encode(value3, t_0=720 * ms))

    recall_spikes = [120, 480, 600, 840] * ms

    spikes_out, statemon = m.run(1000 * ms, spikes_in={
        'store': store_spikes,
        'recall': recall_spikes}, return_statemon=True)

    assert len(spikes_out['output']) == 8

    values_decoded = decode(spikes_out['output'])

    assert isclose(values_decoded[0], value1, atol=abstol)
    assert isclose(values_decoded[1], value2, atol=abstol)

    # recall after memory is depleted
    assert isclose(values_decoded[2], 1., atol=abstol)

    assert isclose(values_decoded[3], value3, atol=abstol)


def test_persistent(value1=0.25, out=None):
    m = Persistent()

    spikes_out, statemon = m.run(300 * ms, spikes_in={'store': encode(value1, t_0=10 * ms),
                                                      'recall': [100, 200] * ms},
                                 return_statemon=True)
    plot_chronogram(statemon, out=out,
                    subset=['store', 'recall', 'vmem0:acc', 'vmem1:acc', 'ready', 'output'])

    assert len(spikes_out['output']) == 4
    values_out = decode(spikes_out['output'])
    assert isclose(values_out, value1, atol=abstol).all()

    # Encode 1 and 0
    # 0,        200       400       600       800       1000      1200      1400      1600
    # encode 1, recall 1, recall 1, recall 1, recall 1, encode 1, encode 0, recall 0, recall 0
    spikes_out, statemon = m.run(1800 * ms, spikes_in={'store': encode(1) +
                                                                encode(1, t_0=1000 * ms) +
                                                                encode(0, t_0=1200 * ms),
                                                       'recall': [200, 400, 600, 800, 1400, 1600] * ms},
                                 return_statemon=True)

    assert len(spikes_out['output']) == 12
    values_out = decode(spikes_out['output'])
    assert isclose(values_out[:4], 1.0, atol=abstol).all()
    assert isclose(values_out[6:], 0.0, atol=abstol).all()


def test_sequential(values=[0.25, 0.15, 0.35], out=None):
    m = Sequential(len(values), spacing=5 * ms)

    spikes_in = {
        'recall': [100] * ms,
    }

    for i, x in enumerate(values):
        spikes_in['store%d' % i] = encode(x, t_0=10 * ms)

    spikes_out, statemon = m.run(300 * ms, spikes_in=spikes_in, return_statemon=True)
    plot_chronogram(statemon, out=out,
                    subset=['vmem0:store', 'vmem1:store', 'vmem2:store',
                            # 'vmem0:acc', 'vmem1:acc', 'vmem2:acc',
                            'recall', 'output'],
                    subplot_labels=['store0', 'store1', 'store2', 'recall', 'output'],
                    show_values=['store0', 'store1', 'store2', 'output'])

    values_decoded = decode(spikes_out['output'], method='sequential')

    assert isclose(values, values_decoded, atol=abstol).all()


def test_superimposed(values=[0.25, 0.15, 0.35], out=None):
    m = Superimposed(len(values))

    spikes_in = {
        'recall': [100] * ms,
    }

    for i, x in enumerate(values):
        spikes_in['store%d' % i] = encode(x, t_0=10 * ms)

    spikes_out, statemon = m.run(300 * ms, spikes_in=spikes_in, return_statemon=True)
    plot_chronogram(statemon, out=out,
                    subset=['vmem0:store', 'vmem1:store', 'vmem2:store',
                            # 'vmem0:acc', 'vmem1:acc', 'vmem2:acc',
                            'recall', 'output'],
                    subplot_labels=['store0', 'store1', 'store2', 'recall', 'output'],
                    show_values={'store0': 'sequential', 'store1': 'sequential', 'store2': 'sequential',
                                 'output': 'superimposed'})

    values_decoded = decode(spikes_out['output'], method='superimposed')

    assert isclose(sort(values), values_decoded, atol=abstol).all()


def test_chained(values=[0.25, 0.15, 0.35], out=None):
    m = Chained(len(values))

    spikes_in = {
        'recall': [100] * ms,
    }

    for i, x in enumerate(values):
        spikes_in['store%d' % i] = encode(x, t_0=10 * ms)

    spikes_out, statemon = m.run(300 * ms, spikes_in=spikes_in, return_statemon=True)
    plot_chronogram(statemon, out=out,
                    subset=['vmem0:store', 'vmem1:store', 'vmem2:store',
                            # 'vmem0:acc', 'vmem1:acc', 'vmem2:acc',
                            'recall', 'output'],
                    subplot_labels=['store0', 'store1', 'store2', 'recall', 'output'])

    values_decoded = decode(spikes_out['output'], method='chained')

    assert isclose(values, values_decoded, atol=abstol).all()


def test_spatial_memory(N=3, out=None):
    m = SpatialMemory(N=N)

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'set0': [10] * ms,
        'set1': [150] * ms,
        'set2': [175] * ms,
        'recall': [50, 200, 250] * ms,
        'reset': [100] * ms,
    }, return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['set0', 'set1', 'set2', 'recall', 'reset', 'out0', 'out1', 'out2'])

    assert len(spikes_out['out0']) == 1
    assert len(spikes_out['out1']) == 2
    assert len(spikes_out['out2']) == 2
    assert spikes_out['out1'][0] == spikes_out['out2'][0]


def test_addressable_memory(N=3, delay=10 * ms, out=None):
    location_value = [
        (0.5 / N, 0.15),
        (1.5 / N, 0.05),
    ]

    locate_spikes, store_spikes, recall_spikes = [], [], []
    next_t_0 = delay
    for location, value in location_value:
        locate_spikes.extend(encode(location, t_0=next_t_0))
        next_t_0 = locate_spikes[-1] + delay
        store_spikes.extend(encode(value, t_0=next_t_0))
        next_t_0 = store_spikes[-1] + delay

    for location, _ in location_value[:1]:
        locate_spikes.extend(encode(location, t_0=next_t_0))
        next_t_0 = locate_spikes[-1] + delay
        recall_spikes.append(next_t_0)
        next_t_0 = recall_spikes[-1] + T_max + delay

    m = AddressableMemory(N=N)

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'locate': locate_spikes,
        'store': store_spikes,
        'recall': recall_spikes,
    }, return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['locate', 'store', 'recall', 'pmem0:vmem0:acc', 'pmem1:vmem0:acc', 'l_ready', 's_ready',
                            'output'])

    # Store 3 values and then recall each location
    location_value = [
        (0.5 / N, 0.35),
        (1.5 / N, 0.25),
        (2.5 / N, 0.60),
    ]

    locate_spikes, store_spikes, recall_spikes = [], [], []
    next_t_0 = 0 * ms
    for location, value in location_value:
        locate_spikes.extend(encode(location, t_0=next_t_0))
        next_t_0 = locate_spikes[-1] + delay
        store_spikes.extend(encode(value, t_0=next_t_0))
        next_t_0 = store_spikes[-1] + delay

    for location, _ in location_value:
        locate_spikes.extend(encode(location, t_0=next_t_0))
        next_t_0 = locate_spikes[-1] + delay
        recall_spikes.append(next_t_0)
        next_t_0 = recall_spikes[-1] + T_max + delay

    m = AddressableMemory(N=N)
    spikes_out, statemon = m.run(1000 * ms, spikes_in={
        'locate': locate_spikes,
        'store': store_spikes,
        'recall': recall_spikes,
    }, return_statemon=True)

    values_out = decode(spikes_out['output'])

    for i, (_, value) in enumerate(location_value):
        assert isclose(values_out[i], value, atol=abstol)


def test_linked_list(delay=5 * ms, out=None):
    values = [0.3, 0.1, 0.2]
    N = len(values)

    next_spikes, store_spikes, recall_spikes = [], [], []
    next_t_0 = delay
    for value in values:
        next_spikes.append(next_t_0)
        next_t_0 = next_spikes[-1] + delay
        store_spikes.extend(encode(value, t_0=next_t_0))
        next_t_0 = store_spikes[-1] + delay

    for value in values:
        next_spikes.append(next_t_0)
        next_t_0 = next_spikes[-1] + delay
        recall_spikes.append(next_t_0)
        next_t_0 = recall_spikes[-1] + encode(value)[1] + delay

    m = LinkedList(N=N)
    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'next': next_spikes,
        'store': store_spikes,
        'recall': recall_spikes,
    }, return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['next', 'store', 'recall', 'output'])

    values_out = decode(spikes_out['output'])

    for i, value in enumerate(values):
        assert isclose(values_out[i], value, atol=abstol)
