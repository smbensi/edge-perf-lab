import time

import numpy as np


def time_function(fn, *args, warmup=3, runs=30, **kwargs):
    if warmup >= runs:
        raise ValueError("The value of runs must be greater than warmup")
    durations = np.zeros(runs - warmup)
    for i in range(runs):
        start = time.perf_counter()  # return float in seconds
        fn(*args, **kwargs)
        end = time.perf_counter()
        if i >= warmup:
            durations[i - warmup] = end - start

    p50 = np.percentile(durations, 50)
    p99 = np.percentile(durations, 99)
    return {"p50_ms": float(p50) * 1e3, "p99_ms": float(p99) * 1e3}
