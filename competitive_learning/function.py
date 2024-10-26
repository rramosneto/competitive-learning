import numpy as np
import numpy.typing as npt


def euclidean_distance(x: npt.NDArray[float], y: npt.NDArray[float]) -> float:
    return np.linalg.norm(x - y)


def fixed_learning_rate(learning_rate: float):
    def learning_rate_function(
        state: int | None = None, n_states: int | None = None
    ) -> float:
        return learning_rate

    return learning_rate_function
