import numpy as np
import pytest

from probability_statistics.classes import Interval, StepFunction
from probability_statistics.survival_analysis import (
    Observation,
    Record,
    compute_new_survival_rate,
    estimate_survival_function,
    init_record,
    record_observations,
    start_estimator,
    update_n_observations_at_risk,
    update_record,
)

observation_0_false = Observation(time=0, censored=False)
observation_0_true = Observation(time=0, censored=True)
observation_1 = Observation(time=1, censored=False)
observation_2 = Observation(time=1, censored=True)
observation_3 = Observation(time=2, censored=False)
observation_4 = Observation(time=3, censored=False)
observation_5 = Observation(time=3, censored=False)
observation_6 = Observation(time=4, censored=True)
observation_7 = Observation(time=5, censored=False)
observations = [
    observation_1,
    observation_2,
    observation_3,
    observation_4,
    observation_5,
    observation_6,
    observation_7,
]

record_1 = Record(time=1, n_censorings=1, n_events=1)
record_2 = Record(time=2, n_censorings=0, n_events=1)
record_3 = Record(time=3, n_censorings=0, n_events=2)
record_4 = Record(time=4, n_censorings=1, n_events=0)
record_5 = Record(time=5, n_censorings=0, n_events=1)
records = [record_1, record_2, record_3, record_4, record_5]

interval_0_0 = Interval(0, 0)
interval_0_1 = Interval(0, 1)
interval_1_1 = Interval(1, 1)
interval_1_2 = Interval(1, 2)
interval_2_3 = Interval(2, 3)
interval_3_5 = Interval(3, 5)
interval_5_5 = Interval(5, 5)

step_0_false = StepFunction(interval_0_0, 1.0)
step_0_true = StepFunction(interval_0_0, 0.0)
step_1 = StepFunction(interval_0_1, 1.0)
step_2 = StepFunction(interval_1_2, 1 * (6 / 7))
step_3 = StepFunction(interval_2_3, 1 * (6 / 7) * (4 / 5))
step_4 = StepFunction(interval_3_5, 1 * (6 / 7) * (4 / 5) * (2 / 4))
step_5 = StepFunction(interval_5_5, 1 * (6 / 7) * (4 / 5) * (2 / 4) * 0)
estimator = [step_1, step_2, step_3, step_4, step_5]


def test_init_record():
    """
    tests init_record()
    """
    # when censored is False, 0 censoring and 1 event
    rec_0 = init_record(observation_1)
    assert (
        rec_0.time == observation_1.time
        and rec_0.n_censorings == 0
        and rec_0.n_events == 1
    )
    # when censored is True, 1 censoring and 0 event
    rec_1 = init_record(observation_2)
    assert (
        rec_1.time == observation_2.time
        and rec_1.n_censorings == 1
        and rec_1.n_events == 0
    )


def test_update_record():
    """
    tests update_record()
    """
    rec_0 = init_record(observation_1)
    # that it raises error when record.time != observation.time
    with pytest.raises(ValueError):
        update_record(rec_0, observation_3)
    # that it updates n_events when observation is not censored
    rec_0 = update_record(rec_0, observation_1)
    assert rec_0.n_events == 2
    # that it updates n_censorings when observation is censored
    rec_0 = update_record(rec_0, observation_2)
    assert rec_0.n_censorings == 1


def test_record_observations():
    """
    tests record_observations()
    """
    # that it records no record for no observation
    assert record_observations([]) == []
    # that it records correctly
    assert record_observations(observations) == records


def test_compute_new_survival_rate():
    """
    tests compute_new_survival_rate()
    """
    # 5 out of 10 survived, now record.n_events = 1 and 4 out of 5 survive. Overall, 4 out of 10 survive
    assert compute_new_survival_rate(record_1, 5, 0.5) == 0.4


def test_update_n_observations_at_risk():
    """
    tests update_n_observations_at_risk()
    """
    # 8 observations at risk after 1 censoring and 1 event out of 10 observations
    assert update_n_observations_at_risk(10, record_1) == 8


def test_start_estimator():
    """
    tests start_estimator()
    """
    steps = start_estimator()
    assert len(steps) == 1
    step = steps[0]
    assert step.support == Interval(0, np.nan) and step.height == 1.0


def test_estimate_survival_function():
    """
    tests estimate_survival_function()
    """
    # that it estimates 0 step for 0 observation
    assert estimate_survival_function([]) == []

    # that it estimates right when there is 1 censoring at time 0
    assert estimate_survival_function([observation_0_true]) == [step_0_false]
    # that it estimates right when there is 1 censoring at time 0 and 1 event at time 1
    assert estimate_survival_function([observation_0_true, observation_1]) == [
        StepFunction(interval_0_1, 1.0),
        StepFunction(interval_1_1, 0.0),
    ]

    # that it estimates right when there is 1 event at time 0
    assert estimate_survival_function([observation_0_false]) == [step_0_true]
    # that it estimates right when there is 1 event at time 0 and 1 censoring at time 1
    assert estimate_survival_function([observation_0_false, observation_2]) == [
        StepFunction(interval_0_1, 0.5),
        StepFunction(interval_1_1, 0.5),
    ]

    # that it estimates right
    assert estimate_survival_function(observations) == estimator
