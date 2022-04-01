import numpy as np
from recordclass import RecordClass


class Interval(RecordClass):
    """
    if left < right then open interval [left, right), else singleton
    """

    left: float
    right: float


class StepFunction(RecordClass):
    """
    step function whose support is support and whose image is constant height
    """

    support: Interval
    height: float

    def __eq__(self, other) -> bool:
        return self.support == other.support and np.isclose(self.height, other.height)
