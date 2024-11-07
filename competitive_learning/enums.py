from enum import Enum

from competitive_learning.function import (
    euclidean_distance,
    exponential_decay,
    fixed_learning_rate,
    linear_decay,
)
from competitive_learning.model import (
    InitializerFactory,
    RandomStrategy,
    WTAStrategy,
    FSCLStrategy,
)


class NeuronInitializer(Enum):
    ZERO_INITIALIZER = "zero_initializer"
    MEAN_INITIALIZER = "mean_initializer"


class LearningRateFunction(Enum):
    CONSTANT = "constant"
    LINEAR = "linear_decay"
    EXPONENTIAL = "exponential_decay"
    # INVERSE = "inverse"
    # SIGMOID = "sigmoid"


class ProximityFunction(Enum):
    EUCLIDEAN_DISTANCE = "euclidean_distance"


class Strategy(Enum):
    RANDOM = "random"
    WTA = "wta"
    FSCL = "fscl"
    # SOM = "som"


INITIALIZER = {
    NeuronInitializer.ZERO_INITIALIZER.value: InitializerFactory.zero_initializer,
    NeuronInitializer.MEAN_INITIALIZER.value: InitializerFactory.mean_initializer,
}

LEARNING_RATE_FUNCTION = {
    LearningRateFunction.CONSTANT.value: fixed_learning_rate,
    LearningRateFunction.LINEAR.value: linear_decay,
    LearningRateFunction.EXPONENTIAL.value: exponential_decay,
}

PROXIMITY_FUNCTION = {ProximityFunction.EUCLIDEAN_DISTANCE.value: euclidean_distance}

STRATEGY = {
    Strategy.RANDOM.value: RandomStrategy,
    Strategy.WTA.value: WTAStrategy,
    Strategy.FSCL.value: FSCLStrategy,
}

available_strategies = [strategy.value for strategy in Strategy]
available_proximity_functions = [
    proximity_function.value for proximity_function in ProximityFunction
]
available_learning_rate_functions = [
    learning_rate_function.value for learning_rate_function in LearningRateFunction
]
available_initializers = [initializer.value for initializer in NeuronInitializer]
