import random

import numpy as np

from anomaly_detection.utils.seeds import set_global_seed


def test_set_global_seed_makes_random_and_numpy_deterministic():
    set_global_seed(42)
    a_py = random.random()
    a_np = float(np.random.rand())

    set_global_seed(42)
    b_py = random.random()
    b_np = float(np.random.rand())

    assert a_py == b_py
    assert a_np == b_np
