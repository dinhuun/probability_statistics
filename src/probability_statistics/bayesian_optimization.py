from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from matplotlib import gridspec
from numpy import ndarray


def my_function(X: ndarray) -> ndarray:
    return np.exp(-((X - 2) ** 2)) + np.exp(-((X - 6) ** 2) / 10) + 1 / (X**2 + 1)


def posterior(
    optimizer: BayesianOptimization, x_obs: ndarray, y_obs: ndarray, grid: ndarray
) -> Tuple[float, float]:
    optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_optimizer(
    optimizer: BayesianOptimization, X: ndarray, Y: ndarray, x_min: float, x_max: float
):
    fig = plt.figure(figsize=(12, 8))
    step_index = len(optimizer.space)
    fig.suptitle(
        f"Gaussian Process and Utility Function After {step_index} Steps",
        fontdict={"size": 30},
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    axis.plot(X, Y, linewidth=3, label="Target")

    x_obs = np.array([[res["params"]["X"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    mu, sigma = posterior(optimizer, x_obs, y_obs, X)
    axis.plot(
        x_obs.flatten(), y_obs, "D", markersize=8, label="Observations", color="r"
    )
    axis.plot(X, mu, "--", color="k", label="Prediction")

    axis.fill(
        np.concatenate([X, X[::-1]]),
        np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=0.6,
        fc="c",
        ec="None",
        label="95% confidence interval",
    )

    axis.set_xlim((x_min, x_max))
    axis.set_ylim((None, None))
    axis.set_ylabel("f(x)", fontdict={"size": 20})
    axis.set_xlabel("x", fontdict={"size": 20})
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(X, optimizer._gp, 0)
    acq = plt.subplot(gs[1])
    acq.plot(X, utility, label="Utility Function", color="purple")
    acq.plot(
        X[np.argmax(utility)],
        np.max(utility),
        "*",
        markersize=15,
        label="Next Best Guess",
        markerfacecolor="gold",
        markeredgecolor="k",
        markeredgewidth=1,
    )
    acq.set_xlim((x_min, x_max))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel("Utility", fontdict={"size": 20})
    acq.set_xlabel("x", fontdict={"size": 20})
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.0)
    plt.show()
