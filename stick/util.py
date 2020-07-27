from stick import *


def encode(x, method='sequential', t_0=0 * ms, spacing=5 * ms, check_bounds=True):
    """Encode real value x to spike time interval"""
    x = asarray(x)

    if check_bounds:
        assert (0 <= x).all()
        assert (x <= 1).all()

    intervals = T_min + x * T_cod

    if method == 'sequential':
        if x.ndim > 0:
            offsets = merge(0 * ms, spacing.repeat(len(x) - 1).cumsum())
            second_spikes = t_0 + intervals.cumsum() + offsets.cumsum()
        else:
            second_spikes = t_0 + intervals

        t = merge(second_spikes - intervals, second_spikes)
    elif method == 'chained':
        t = merge(t_0, intervals).cumsum()
    elif method == 'superimposed':
        t = merge(t_0, t_0 + intervals)
    else:
        raise Exception('Unknown encoding method: %s' % str(method))

    return t


def decode(t, method='sequential', check_bounds=True):
    """Decode a sequence of spike time intervals to real value x"""

    t = asarray(t / ms) * ms

    if method == 'sequential':
        assert (len(t) % 2) == 0
        t_first, t_second = t[::2], t[1::2]
    elif method == 'chained':
        t_first, t_second = t[:-1], t[1:]
    elif method == 'superimposed':
        t_first = t[0].repeat(len(t) - 1)
        t_second = t[1:]
    else:
        raise Exception('Unknown decoding method: %s' % str(method))

    delta_t = t_second - t_first

    if check_bounds:
        # Within dt of T_min and T_max
        assert all(T_min - dt / 2 <= delta_t) and all(delta_t <= T_max + dt / 2)

    return (delta_t - T_min) / T_cod


def merge(*args):
    """Merge multiple arrays of spike times"""

    t = []
    for a in args:
        if a.ndim > 0:
            t.extend(a / ms)
        else:
            t.append(a / ms)

    return sort(asarray(t)) * ms
