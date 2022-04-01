"""
developed with Michael T. Wojnowicz"
"""

from typing import List, NamedTuple, Tuple

import numpy as np
from recordclass import RecordClass

from probability_statistics.classes import Interval, StepFunction

KaplanMeierEstimator = List[StepFunction]  # with increasing consecutive supports


class Observation(NamedTuple):
    """
    observation of one individual
    attributes:
        * time: how long observed individual survived, either to some time or to the end of observation
        * censored: true if no event happened during observation and individual survived it
    """

    time: float
    censored: bool


class Record(RecordClass):
    """
    records number of events/censorings at time
    attributes:
        * time: time at which events/censorings happen
        * n_censorings: number of censorings that happen
        * n_events: number of events that happen
    """

    time: float
    n_censorings: int
    n_events: int


def init_record(observation: Observation) -> Record:
    """
    initializes record for observation
    """
    if observation.censored:
        return Record(observation.time, 1, 0)
    else:
        return Record(observation.time, 0, 1)


def update_record(record: Record, observation: Observation) -> Record:
    """
    updates record to new observation of same time
    """
    if record.time != observation.time:
        raise ValueError(
            f"record {record} and observation {observation} have different times"
        )

    if observation.censored:
        record.n_censorings += 1
    else:
        record.n_events += 1
    return record


def record_observations(observations: List[Observation]) -> List[Record]:
    """
    records observations into records
    """
    if observations:
        observations = sorted(observations, key=lambda obs: obs.time)
        records = []
        zeroth_observation = observations[0]
        prev_time = zeroth_observation.time
        record = init_record(zeroth_observation)
        for observation in observations[1:]:
            curr_time = observation.time
            if curr_time == prev_time:
                record = update_record(record, observation)
            else:
                records.append(record)
                record = init_record(observation)
            prev_time = curr_time
        records.append(record)
        return records
    else:
        return []


def compute_new_survival_rate(
    record: Record, n_observations_at_risk: int, prev_survival_rate: float
) -> float:
    """
    computes new survival rate. It is prev_survival_rate * (1 - event_rate)
    :param record: record
    :param n_observations_at_risk: number of observations at risk at record time
    :param prev_survival_rate: previous survival rate
    :return: new survival rate
    """
    event_rate = record.n_events / n_observations_at_risk
    return prev_survival_rate * (1 - event_rate)


def update_n_observations_at_risk(n_observations_at_risk: int, record: Record) -> int:
    """
    updates number of observations at risk by dropping censorings and events
    :param n_observations_at_risk: number of observations at risk
    :param record: record with number of censorings and number of events
    :return: new number of observations at risk
    """
    n_observations_at_risk -= record.n_censorings + record.n_events
    return n_observations_at_risk


def start_estimator() -> KaplanMeierEstimator:
    """
    starts estimator off with step at time 0, height 1.0
    :return:
    """
    support = Interval(0, np.nan)
    step = StepFunction(support, 1.0)
    return [step]


def extend_estimator(
    estimator: KaplanMeierEstimator, record: Record, n_observations_at_risk: int
) -> KaplanMeierEstimator:
    """
    extends estimator 1 more step
    :param estimator: estimator
    :param record: record
    :param n_observations_at_risk: number of observations at risk
    :return: extended estimator
    """
    # censorings/events at time 0, restart zeroth step
    if record.time == 0:
        support = Interval(0, np.nan)
        height = compute_new_survival_rate(record, n_observations_at_risk, 1.0)
        step = StepFunction(support, height)
        return [step]
    # no events, no new step
    elif record.n_events == 0:
        return estimator
    # events, new step
    else:
        prev_step = estimator[-1]
        prev_step.support.right = record.time
        support = Interval(record.time, np.nan)
        height = compute_new_survival_rate(
            record, n_observations_at_risk, prev_step.height
        )
        next_step = StepFunction(support, height)
        estimator.append(next_step)
        return estimator


def end_estimator(
    estimator: KaplanMeierEstimator, record: Record, n_observations_at_risk: int
) -> KaplanMeierEstimator:
    """
    ends estimator with last step
    :param estimator: estimator
    :param record: record
    :param n_observations_at_risk: number of observations at risk
    :return: estimator
    """
    prev_step = estimator[-1]
    right = record.time
    prev_step.support.right = right

    support = Interval(right, right)
    if n_observations_at_risk == 0:
        height = prev_step.height
    else:
        height = compute_new_survival_rate(
            record, n_observations_at_risk, prev_step.height
        )
    last_step = StepFunction(support, height)

    if right == 0:
        return [last_step]
    else:
        estimator.append(last_step)
        return estimator


def estimate_survival_function(
    observations: List[Observation],
) -> KaplanMeierEstimator:
    """
    estimates survival function by step functions
    :param observations: observations
    :return: steps
    """
    n_observations_at_risk = len(observations)
    if n_observations_at_risk == 0:
        return []

    records = record_observations(observations)

    estimator = start_estimator()
    for record in records[:-1]:
        estimator = extend_estimator(estimator, record, n_observations_at_risk)
        n_observations_at_risk = update_n_observations_at_risk(
            n_observations_at_risk, record
        )
    estimator = end_estimator(estimator, records[-1], n_observations_at_risk)
    return estimator


def get_estimator_coordinates(
    estimator: KaplanMeierEstimator,
) -> Tuple[List[float], List[float]]:
    """
    gets KaplanMeierEstimator steps' supports and heights for plotting
    :param estimator: estimator
    :return: steps' supports and heights for plotting
    """
    zeroth_step = estimator[0]
    x_s = [zeroth_step.support.left]
    y_s = [zeroth_step.height]
    for step in estimator:
        x_s.append(step.support.right)
        y_s.append(step.height)
    return x_s, y_s
