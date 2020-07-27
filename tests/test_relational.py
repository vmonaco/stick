from stick import *


def test_minimum(smaller=0.1, larger=0.3):
    m = Minimum()
    spikes_out, statemon = m.run(300 * ms, spikes_in={'input1': encode(smaller, t_0=10 * ms),
                                                      'input2': encode(larger, t_0=10 * ms)},
                                 return_statemon=True)
    # plot_chronogram(statemon, out=os.path.join(FIGURES_DIR, 'minimum.pdf'))

    values_out = decode(spikes_out['output'])
    assert isclose(values_out, smaller, atol=abstol).all()
    assert len(spikes_out['smaller1']) == 1 and len(spikes_out['smaller2']) == 0

    # Reverse the inputs
    spikes_out = m.run(300 * ms, spikes_in={'input1': encode(larger, t_0=10 * ms),
                                            'input2': encode(smaller, t_0=10 * ms)})
    values_out = decode(spikes_out['output'])
    assert isclose(values_out, smaller, atol=abstol).all()
    assert len(spikes_out['smaller1']) == 0 and len(spikes_out['smaller2']) == 1


def test_maximum(smaller=0.1, larger=0.3):
    m = Maximum()
    spikes_out, statemon = m.run(300 * ms, spikes_in={'input1': encode(smaller, t_0=10 * ms),
                                                      'input2': encode(larger, t_0=10 * ms)},
                                 return_statemon=True)
    # plot_chronogram(statemon, out=os.path.join(FIGURES_DIR, 'maximum.pdf'))

    values_out = decode(spikes_out['output'])
    assert isclose(values_out, larger, atol=abstol).all()
    assert len(spikes_out['larger1']) == 0 and len(spikes_out['larger2']) == 1

    # Reverse the inputs
    spikes_out = m.run(300 * ms, spikes_in={'input1': encode(larger, t_0=10 * ms),
                                            'input2': encode(smaller, t_0=10 * ms)})

    values_out = decode(spikes_out['output'])
    assert isclose(values_out, larger, atol=abstol).all()
    assert len(spikes_out['larger1']) == 1 and len(spikes_out['larger2']) == 0
