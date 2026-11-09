import time

import pytest

from bench.timer import time_function


def test_sleep_10ms_is_measured_around_10ms():
    result = time_function(time.sleep, 0.01, warmup=2, runs=20)
    assert 9 <= result["p50_ms"] <= 100
    assert result["p99_ms"] >= result["p50_ms"]


@pytest.mark.parametrize("warmup,runs", [(-2, 4), (12, -3), (0,0)])
def test_reject_negative_args(warmup, runs):
    with pytest.raises(ValueError):
        time_function(lambda: None, warmup=warmup, runs=runs)
