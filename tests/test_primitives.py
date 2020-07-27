from stick import *

def test_ge_synapse(out=None):
    ge1 = GeSynapse(1)

    spikes_out, statemon = ge1.run(300 * ms,
                                 spikes_in={'pre':[20, 30.1]*ms},
                                 return_statemon=True)

    plot_chronogram(statemon, out=out+'/simulate-ge-chronogram.pdf')

    ge3 = GeSynapse(2)

    spikes_out, statemon = ge3.run(300 * ms,
                                 spikes_in={'pre':[20, 40.20]*ms},
                                 return_statemon=True)
    
    plot_chronogram(statemon, out=out+'/simulate-ge-multi-chronogram.pdf')


def test_iterator(out=None):
    values = [0.3, 0.1, 0.2]
    N = len(values)

    m = Iterator(N=N)

    spikes_out, statemon = m.run(300 * ms, spikes_in={
        'next': [10, 50, 90, 110, 160, 220] * ms,
    }, return_statemon=True)

    plot_chronogram(statemon, out=out)

    assert len(spikes_out['addr0']) == 2
    assert len(spikes_out['addr1']) == 2
    assert len(spikes_out['addr2']) == 2


def test_addresser(N=5, out=None):
    m = Addresser(N=3)
    spikes_out, statemon = m.run(300 * ms,
                                 spikes_in={'locate': encode([0.5 / 3, 2.5 / 3], t_0=10 * ms, spacing=20 * ms)},
                                 return_statemon=True)
    plot_chronogram(statemon, out=out,
                    subset=['locate', 'acc0', 'acc1', 'acc2', 'addr0', 'addr1', 'addr2'])

    m = Addresser(N=N)
    for i in range(N):
        spikes_in = {
            'locate': encode((i + 0.5) / N)
        }

    spikes_out, statemon = m.run(200 * ms, spikes_in=spikes_in, return_statemon=True)

    for j in range(N):
        try:
            if i == j:
                assert len(spikes_out['addr%d' % j]) > 0
            else:
                assert len(spikes_out['addr%d' % j]) == 0
        except:
            from IPython import embed
            embed()
            raise Exception()
