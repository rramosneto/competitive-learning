import numpy as np
import numpy.typing as npt
from functools import partial


def euclidean_distance(x: npt.NDArray[float], y: npt.NDArray[float]) -> float:
    return np.linalg.norm(x - y)


def fixed_learning_rate(learning_rate: float, n_states: int):
    return partial(
        _fixed_learning_rate_function, learning_rate=learning_rate, n_states=n_states
    )


def _fixed_learning_rate_function(
    state: int | None = None, n_states: int | None = None, learning_rate: float = 0.0
) -> float:
    return learning_rate


def linear_decay(learning_rate: float, n_states: int):
    return partial(
        _linear_decay_function, learning_rate=learning_rate, n_states=n_states
    )


def _linear_decay_function(
    state: int | None = None, n_states: int | None = None, learning_rate: float = 0.0
) -> float:
    if state is None or n_states is None:
        raise ValueError("State and n_states must be provided for linear decay.")
    return learning_rate * (1 - min(state, n_states) / n_states)


def exponential_decay(learning_rate: float, n_states: int):
    return partial(
        _exponential_decay_function, learning_rate=learning_rate, n_states=n_states
    )


def _exponential_decay_function(
    state: int | None = None,
    n_states: int | None = None,
    learning_rate: float = 0.0,
    final_rate: float = 0.001,
) -> float:
    if state is None or n_states is None:
        raise ValueError("State and n_states must be provided for exponential decay.")
    return learning_rate * (final_rate / learning_rate) ** (
        min(state, n_states) / n_states
    )
