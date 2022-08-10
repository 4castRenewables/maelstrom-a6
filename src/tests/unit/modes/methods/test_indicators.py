import lifetimes.modes.methods.indicators as indicators
import numpy as np


def test_indicators():
    data = np.random.randn(2, 10)
    indicators(
        data,
        Y=None,
        metric="euclidean",
        q=0.97,
        pareto_fit="scipy",
        theta_fit="sueveges",
        distances=None,
    )
