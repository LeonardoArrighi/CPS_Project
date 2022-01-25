import numpy as np


def overshoot(x, ref):
    """Difference between the max value of the system output and the desired reference value"""
    over = round(max(x) - ref, 3)
    return over


def rise_time(x, t, ref):
    """Time point at which the output signal crosses the desired reference value"""
    ref = np.array([ref for i in range(len(x))])
    idx = min(np.where(np.abs(x - ref) < 1))[0]
    rise = t[idx]
    return rise


def steady_state_error(x, ref):
    """Difference between steady state value of the output signal and value of the reference signal"""
    error = round(np.abs(x[-1] - ref), 3)
    return error


# def settling_time(x, t):
#     """Time at which the output reaches its steady state value"""
#     idx = np.where(np.abs(x - x[-1]) < 0.5)[0]  # put 1 as parameters
#     idx = [j for i, j in enumerate(idx) if abs(j - idx[min(i + 1, len(idx) - 1)]) < 2]
#     i = min(idx)
#     time = t[i]
#     return time

def settling_time(x, t):
    """Time at which the output reaches its steady state value"""
    stop = int(len(x)/10)
    for i in range(len(x) - stop):
        sum = 0
        for j in range(stop):
            sum += np.abs(x[i] - x[i+j])
        if sum < 16:
            time = t[i]
            break
    return time
    