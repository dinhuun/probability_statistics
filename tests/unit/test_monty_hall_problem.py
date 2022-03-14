import numpy as np

from probability_statistics.monty_hall_problem import play


def test_play():
    """
    tests play()
    """
    n_games = 1000
    _, wins_by_staying, wins_by_switching = play(n_games)
    np.testing.assert_almost_equal(wins_by_staying / n_games, 0.33, decimal=2)
    np.testing.assert_almost_equal(wins_by_switching / n_games, 0.66, decimal=2)
