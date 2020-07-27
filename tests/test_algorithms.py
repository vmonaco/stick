from stick import *


def test_sorting_v1(delay=10 * ms, out=None):
    values = [0.1, 0, 0.15, 0.05]
    N = len(values)
    m = Sorting_v1(N=N)

    store_spikes = []
    next_t_0 = delay
    for value in values:
        store_spikes.extend(encode(value, t_0=next_t_0))
        next_t_0 = store_spikes[-1] + delay

    sort_spikes = [store_spikes[-1] + delay]
    recall_spikes = [sort_spikes[-1] + 50 * ms]

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'input': store_spikes,
        'sort': sort_spikes,
        'recall': recall_spikes
    }, return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['input', 'sort', 'recall', 'input_list:output', 'store_ready', 'sort_ready', 'recall_ready',
                            'output'])

    assert len(spikes_out['output']) == len(values) * 2
    values_out = decode(spikes_out['output'])

    for value, sorted_value in zip(values_out, sorted(values)):
        assert isclose(value, sorted_value, atol=abstol)


def test_sorting(delay=20 * ms, out=None):
    values = [0.25, 0, 0.4]
    N = len(values)
    m = Sorting(N=N)

    spikes_in = {}
    for i, x in enumerate(values):
        spikes_in['store%d' % i] = encode(x)

    spikes_in['sort'] = [max(encode(max(values))) + delay]
    spikes_in['recall'] = [2 * max(encode(max(values))) + 2 * delay]

    spikes_out, statemon = m.run(300 * ms, spikes_in=spikes_in, return_statemon=True)

    plot_chronogram(statemon, out=out,
                    subset=['memory_in:vmem0:store', 'memory_in:vmem1:store', 'memory_in:vmem2:store',
                            'sort', 'memory_out:recall', 'ready', 'memory_out:output'
                            # 'router:out0', 'router:out1', 'router:out2', 'router:out3',
                            # 'memory_out:vmem0:store', 'memory_out:vmem1:store', 'memory_out:vmem2:store',
                            # 'memory_out:recall',
                            ], subplot_labels=['store0', 'store1', 'store2', 'sort', 'recall', 'ready', 'output'],
                    show_values=['store0', 'store1', 'store2', 'output'])

    assert len(spikes_out['output']) == len(values) * 2
    values_out = decode(spikes_out['output'])

    for value, sorted_value in zip(values_out, sorted(values)):
        assert isclose(value, sorted_value, atol=abstol)


def test_searching(delay=20 * ms, out=None):
    def test_searching_aux(search, values, out=None):
        N = len(values)
        m = Searching(N=N)

        spikes_in = {}
        for i, x in enumerate(values):
            spikes_in['store%d' % i] = encode(x)

        spikes_in['search'] = encode(search, t_0=[max(encode(max(values))) + delay])

        spikes_out, statemon = m.run(350 * ms, spikes_in=spikes_in, return_statemon=True)

        print('Value set:', values)
        print('Searching for:', search)
        print('Spikes out:', spikes_out['output'])
        try:
            print('Values decoded:', decode(spikes_out['output']))
        except:
            print('Failed to decode spikes.')

        if out:
            plot_chronogram(statemon, out=out,
                            subset=['search',
                                    # 'first',
                                    # 'last',
                                    # 'memory:recall',
                                    'memory:output',
                                    # 'router:next',
                                    # 'router:input',
                                    # 'router:out0',
                                    # 'router:out1',
                                    # 'router:out2',
                                    # 'router:out3',
                                    'acc',
                                    'acc_diff',
                                    'acc_max',
                                    'race:output',
                                    ],
                            subplot_labels=['search', 'memory:output', 'acc', 'acc_diff', 'acc_max', 'output'],
                            show_values={'search': 'sequential',
                                         'memory:output': 'superimposed',
                                         'output': 'sequential'})

        assert len(spikes_out['output']) == 2
        values_decoded = decode(spikes_out['output'])
        closest_value_idx = np.argmin(np.abs(values_decoded - values))
        assert isclose(values_decoded, values[closest_value_idx])

    # Greater than
    test_searching_aux(0.35, [0.1, 0.3, 0.6], out=out[:-4] + '-greater.pdf')
    test_searching_aux(0.5, [0.1, 0.3, 0.6], out=out[:-4] + '-less.pdf')

    ## Tests

    values = [0.3, 0.5, 0.9]

    # Slightly greater
    print('# Test slightly greater')
    test_searching_aux(0.5001, values)
    print()

    # Slightly less
    print('# Test slightly less')
    test_searching_aux(0.8999, values)
    print()

    # Slightly greater than halfway
    print('# Test slightly greater than halfway')
    test_searching_aux(0.4001, values)
    print()

    # Slightly less than halfway
    print('# Test slightly less than halfway')
    test_searching_aux(0.3999, values)
    print()

    # Greater than
    print('# Test greater')
    test_searching_aux(0.35, values)
    print()

    # Exactly halfway
    print('# Test exactly halfway')
    test_searching_aux(0.4, values)
    print()

    # Less than
    print('# Test less')
    test_searching_aux(0.8, values)
    print()

    # Equal
    print('# Test equal')
    test_searching_aux(0.5, values)
    print()
