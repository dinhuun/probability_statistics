import numpy as np

from probability_statistics.coin_factory_problem import (
    generate_posterior_samples,
    select_posterior_samples,
)


def test_generate_posterior_samples():
    """
    tests generate_posterior_samples()
    """
    n_tosses = 100
    n_heads = 50
    data = (n_tosses, n_heads)
    n_samples = 1000
    samples = generate_posterior_samples(data, n_samples)
    assert len(samples) == n_samples
    np.testing.assert_almost_equal(n_heads / n_tosses, np.mean(samples), decimal=2)


def test_select_posterior_samples():
    """
    tests select_posterior_samples()
    """
    samples = list(range(10))
    burn_in = 2
    lag = 3
    samples = select_posterior_samples(samples, burn_in, lag)
    assert samples == [2, 5, 8]
