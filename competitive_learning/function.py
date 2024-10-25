import numpy as np
import numpy.typing as npt


def euclidean_distance(x: npt.NDArray[float], y: npt.NDArray[float]) -> float:
    return np.linalg.norm(x - y)


def fixed_learning_rate(learning_rate: float) -> float:
    return learning_rate


# def euclidean_distance(x, y):
#     return sum([(x[i] - y[i]) ** 2 for i in range(len(x))]) ** 0.5

# def manhattan_distance(x, y):
#     return sum([abs(x[i] - y[i]) for i in range(len(x))])

# def chebyshev_distance(x, y):
#     return max([abs(x[i] - y[i]) for i in range(len(x))])

# def hamming_distance(x, y):
#     return sum([x[i] != y[i] for i in range(len(x))])

# def cosine_similarity(x, y):
#     return sum([x[i] * y[i] for i in range(len(x))]) / (sum([x[i] ** 2 for i in range(len(x))]) ** 0.5 * sum([y[i] ** 2 for i in range(len(x))]) ** 0.5)

# def minkowski_distance(x, y, p):
#     return sum([abs(x[i] - y[i]) ** p for i in range(len(x))] ** (1 / p))
