from abc import ABC, abstractmethod
from random import random
from typing import Any, Callable, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal


class DataPoint(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    coordinates: Tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    @property
    def norm(self) -> float:
        return sum(coord**2 for coord in self.coordinates) ** 0.5

    def __getitem__(self, index: int) -> float:
        try:
            return self.coordinates[index]
        except IndexError:
            raise IndexError(
                f"Index {index} is out of range for coordinates with length {len(self.coordinates)}"
            )


class Dataset(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    data_points: Tuple[DataPoint, ...]
    available_points: Set[int] = Field(default_factory=set)
    used_points: Set[int] = Field(default_factory=set)
    state: int = 0
    reset_count: int = 0

    def model_post_init(self, __context) -> None:
        self.available_points = set(range(len(self.data_points)))

    @property
    def dimension(self) -> int:
        return self.data_points[0].dimension

    @property
    def len(self) -> int:
        return len(self.data_points)

    def reset(self) -> None:
        self.reset_count += 1
        self.state = 0
        self.used_points.clear()
        self.available_points = set(range(len(self.data_points)))

    def __getitem__(self, index: int) -> DataPoint:
        if index in self.used_points:
            raise ValueError(f"DataPoint with index {index} has already been used")
        try:
            self.used_points.add(index)
            self.available_points.remove(index)
            return self.data_points[index]
        except IndexError:
            raise IndexError(
                f"Index {index} is out of range for data_points with length {len(self.data_points)}"
            )

    @model_validator(mode="before")
    @classmethod
    def check_data_points_dimensions(cls, data: Any) -> Any:
        data_points = data.get("data_points", [])
        if data_points:
            first_dimension = data_points[0].dimension
            for dp in data_points:
                if dp.dimension != first_dimension:
                    raise ValueError("All data points must have the same dimension")
        return data

    @classmethod
    def from_csv(cls, csv_path: str) -> "Dataset":
        data_points = []
        with open(csv_path) as f:
            for line in f:
                data_points.append(
                    DataPoint(coordinates=tuple(map(float, line.strip().split(","))))
                )
        return cls(data_points=tuple(data_points))


class Neuron(DataPoint):
    iterations: int = 0

    def update_weights(self, coordinates: Tuple[int, ...]) -> None:
        self.iterations += 1
        self.coordinates = coordinates


class NeuralNetwork(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    neurons: Tuple[Neuron, ...]
    state: int = 0
    epoch: int = 0

    @property
    def dimension(self) -> int:
        return self.neurons[0].dimension

    @property
    def len(self) -> int:
        return len(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        try:
            return self.neurons[index]
        except IndexError:
            raise IndexError(
                f"Index {index} is out of range for neurons with length {len(self.neurons)}"
            )

    @model_validator(mode="before")
    @classmethod
    def check_neurons_dimensions(cls, data: Any) -> Any:
        neurons = data.get("neurons", [])
        if neurons:
            first_dimension = neurons[0].dimension
            for n in neurons:
                if n.dimension != first_dimension:
                    raise ValueError("All neurons must have the same dimension")
        return data


class LearningStrategy(BaseModel, ABC):
    id: UUID = Field(default_factory=uuid4)
    proximity_function: Callable
    learning_rate: Callable

    @abstractmethod
    def choose_neurons(self, data_point: DataPoint, neurons: Tuple[Neuron, ...]) -> Tuple[Neuron, ...]:
        pass

    @abstractmethod
    def uptate_neurons_weights(self, neurons: Tuple[Neuron, ...]) -> None:
        pass


class DummyStrategy(LearningStrategy):
    def choose_neurons(self, data_point: DataPoint, neurons: Tuple[Neuron, ...]) -> Tuple[Neuron, ...]:
        return neurons[0]

    def uptate_neurons_weights(self, neurons: Tuple[Neuron, ...]) -> None:
        for neuron in neurons:



class WTAStrategy(LearningStrategy): ...


class FSCLStrategy(LearningStrategy): ...


class SOMStrategy(LearningStrategy): ...


class Experiment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    learning_strategy: LearningStrategy
    neural_network: NeuralNetwork
    dataset: Dataset
    n_epochs: int


class ExperimentFactory(BaseModel):
    @staticmethod
    def create_experiment(
        learning_strategy: LearningStrategy,
        neural_network_initializer: Literal["random", "zero"],
        n_neurons: int,
        dataset: Dataset,
        n_epochs: int,
    ) -> Experiment:
        dimension = dataset.dimension

        if neural_network_initializer == "random":
            neural_network = NeuralNetwork(
                neurons=[
                    Neuron(coordinates=tuple([random() for _ in range(dimension)]))
                    for _ in range(n_neurons)
                ]
            )
        elif neural_network_initializer == "zero":
            neural_network = NeuralNetwork(
                neurons=[
                    Neuron(coordinates=tuple([0 for _ in range(dimension)]))
                    for _ in range(n_neurons)
                ]
            )

        return Experiment(
            learning_strategy=learning_strategy,
            neural_network=neural_network,
            dataset=dataset,
            n_epochs=n_epochs,
        )
