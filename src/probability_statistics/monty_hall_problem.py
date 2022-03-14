import random
from typing import Tuple

import pandas as pd
from pandas import DataFrame

pd.set_option('display.precision', 0)


def play(n_games: int = 1000, seed: int = 0) -> Tuple[DataFrame, int, int]:
    """
    plays a few Monty Hall games
    :param n_games: number of games to play
    :param seed: random seed
    :return: dataframe of games, number of wins by staying, number of wins by switching
    """
    columns = ["prize is at", "player guesses", "host opens", "player switches"]
    doors = {1, 2, 3}
    games = pd.DataFrame(columns=columns)
    random.seed(seed)
    wins_by_staying = 0
    wins_by_switching = 0

    for i in range(n_games):
        prize_is_at = random.randint(1, 3)
        player_guesses = random.randint(1, 3)
        host_opens = (doors - {prize_is_at, player_guesses}).pop()
        remaining_door = (doors - {player_guesses, host_opens}).pop()
        games.loc[i] = [prize_is_at, player_guesses, host_opens, remaining_door]
        if prize_is_at == player_guesses:
            wins_by_staying += 1
        else:
            wins_by_switching += 1

    return games, wins_by_staying, wins_by_switching
