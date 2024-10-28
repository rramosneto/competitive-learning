from enum import Enum

from competitive_learning.function import euclidean_distance, fixed_learning_rate
from competitive_learning.model import InitializerFactory, RandomStrategy, WTAStrategy


class NeuronInitializer(Enum):
    ZERO_INITIALIZER = "zero_initializer"
    MEAN_INITIALIZER = "mean_initializer"


class LearningRateFunction(Enum):
    CONSTANT = "constant"
    # LINEAR = "linear"
    # EXPONENTIAL = "exponential"
    # INVERSE = "inverse"
    # SIGMOID = "sigmoid"


class ProximityFunction(Enum):
    EUCLIDEAN_DISTANCE = "euclidean_distance"


class Strategy(Enum):
    RANDOM = "random"
    WTA = "wta"
    # FSCL = "fscl"
    # SOM = "som"


INITIALIZER = {
    NeuronInitializer.ZERO_INITIALIZER.value: InitializerFactory.zero_initializer,
    NeuronInitializer.MEAN_INITIALIZER.value: InitializerFactory.mean_initializer,
}

LEARNING_RATE_FUNCTION = {LearningRateFunction.CONSTANT.value: fixed_learning_rate}

PROXIMITY_FUNCTION = {ProximityFunction.EUCLIDEAN_DISTANCE.value: euclidean_distance}

STRATEGY = {Strategy.RANDOM.value: RandomStrategy, Strategy.WTA.value: WTAStrategy}
