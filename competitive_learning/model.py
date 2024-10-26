from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Set, Tuple
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator
from pydantic.config import ConfigDict  # noqa: F401

from competitive_learning.function import euclidean_distance, fixed_learning_rate


class DataPoint(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    coordinates: npt.NDArray[np.float64]

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.coordinates)

    def __getitem__(self, index: int) -> float:
        try:
            return self.coordinates[index]
        except IndexError:
            raise IndexError(
                f"Index {index} is out of range for coordinates with length {len(self.coordinates)}"
            )

    @classmethod
    def dummy(cls, n: int = 2) -> "DataPoint":
        return cls(coordinates=np.array([random.random() for _ in range(n)]))

    class Config:
        arbitrary_types_allowed = True


class Dataset(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    data_points: Tuple[DataPoint, ...]
    available_points: Set[int] = Field(default_factory=set)
    used_points: Set[int] = Field(default_factory=set)
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
        self.used_points.clear()
        self.available_points = set(range(len(self.data_points)))

    def get_one(self) -> DataPoint:
        index = random.choice(list(self.available_points))
        return self[index]

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
                    DataPoint(
                        coordinates=np.array(tuple(map(float, line.strip().split(","))))
                    )
                )
        return cls(data_points=tuple(data_points))

    @classmethod
    def dummy(cls, dimension, len) -> "Dataset":
        return cls(data_points=tuple(DataPoint.dummy(dimension) for _ in range(len)))


class Neuron(DataPoint):
    iterations: int = 0

    def update_weights(self, coordinates: Tuple[float, ...]) -> None:
        self.iterations += 1
        self.coordinates = coordinates


class NeuralNetwork(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    neurons: Tuple[Neuron, ...]

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

    @classmethod
    def dummy(cls, dimension: int) -> "NeuralNetwork":
        return cls(neurons=(Neuron.dummy(dimension), Neuron.dummy(dimension)))


class LearningStrategy(BaseModel, ABC):
    id: UUID = Field(default_factory=uuid4)
    proximity_function: Callable
    neural_network: NeuralNetwork

    @abstractmethod
    def choose_neurons(self, data_point: DataPoint) -> Tuple[Tuple[Neuron, ...], float]:
        pass

    @abstractmethod
    def uptate_neurons_weights(
        self,
        data_point: DataPoint,
        neurons: Tuple[Neuron, ...],
        learning_rate: float,
    ) -> None:
        pass

    @classmethod
    def dummy(cls) -> "LearningStrategy":
        return RandomStrategy(proximity_function=lambda x: x, learning_rate=lambda x: x)


class RandomStrategy(LearningStrategy):
    def choose_neurons(self, data_point: DataPoint) -> Tuple[Tuple[Neuron, ...], float]:
        neuron = random.choice(self.neural_network.neurons)
        distance = self.proximity_function(data_point.coordinates, neuron.coordinates)

        return (neuron,), distance

    def uptate_neurons_weights(
        self,
        data_point: DataPoint,
        neurons: Tuple[Neuron, ...],
        learning_rate: float,
    ) -> None:
        for neuron in neurons:
            coordinates = neuron.coordinates + learning_rate * (
                data_point.coordinates - neuron.coordinates
            )
            neuron.update_weights(coordinates=coordinates)

    @classmethod
    def dummy(cls) -> "RandomStrategy":
        return cls(
            proximity_function=euclidean_distance,
            learning_rate=fixed_learning_rate(0.2),
        )


class StateVector(BaseModel):
    n: List[int] = []
    epoch: List[int] = []
    iteration: List[int] = []
    active_data_point: List[UUID] = []
    active_neurons: List[List[UUID]] = []
    learning_rate: List[float] = []
    quantization_error: List[float] = []

    def update(
        self,
        step: int,
        epoch: int,
        iteration: int,
        active_data_point: UUID,
        active_neurons: List[UUID],
        learning_rate: float,
        quantization_error: float,
    ) -> None:
        self.n.append(step)
        self.epoch.append(epoch)
        self.iteration.append(iteration)
        self.active_data_point.append(active_data_point)
        self.active_neurons.append(active_neurons)
        self.learning_rate.append(learning_rate)
        self.quantization_error.append(quantization_error)


class Experiment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    learning_strategy: LearningStrategy
    learning_rate: Callable
    dataset: Dataset
    n_epochs: int
    step: int = 0
    state: int = 0
    epoch: int = 0
    state_vector: StateVector = Field(default_factory=StateVector)

    def run_experiment(self) -> None:
        while self.epoch < self.n_epochs:
            self.run_epoch()
        return

    def run_to_end(self) -> None:
        while self.step < self.n_states:
            self.run_step()

    def run_epoch(self) -> None:
        while len(self.dataset.used_points) < self.dataset.len:
            if self.step == self.n_states:
                break
            self.run_step()
        self.run_step()
        return

    def run_n_steps(self, n: int) -> None:
        for _ in range(n):
            self.run_step()

    def run_step(self) -> None:
        if (
            self.state < self.dataset.len - 1
            and self.dataset.reset_count < self.n_epochs
        ):
            self._run()

        elif (
            self.state == self.dataset.len - 1
            and self.dataset.reset_count < self.n_epochs
        ):
            self._run()
            self.epoch += 1
            self.state = 0
            self.dataset.reset()

        else:
            return

    def _run(self) -> None:
        data_point = self.dataset.get_one()
        neurons, distance = self.learning_strategy.choose_neurons(data_point=data_point)
        learning_rate = self.learning_rate(state=self.step)
        self.learning_strategy.uptate_neurons_weights(
            data_point=data_point,
            neurons=neurons,
            learning_rate=learning_rate,
        )
        self.state_vector.update(
            step=self.step,
            epoch=self.epoch,
            iteration=self.state,
            active_data_point=data_point.id,
            active_neurons=[n.id for n in neurons],
            learning_rate=learning_rate,
            quantization_error=distance,
        )
        self.step += 1
        self.state += 1

    @property
    def n_states(self) -> int:
        return self.n_epochs * self.dataset.len


class InitializerFactory(BaseModel):
    @staticmethod
    def zero_initializer(n_neurons: int, n_dimensions: int) -> NeuralNetwork:
        return NeuralNetwork(
            neurons=tuple(
                [Neuron(coordinates=np.zeros(n_dimensions)) for _ in range(n_neurons)]
            )
        )


class ExperimentFactory(BaseModel):
    @staticmethod
    def create_experiment(
        dataset: Dataset,
        learning_strategy: LearningStrategy,
        n_neurons: int,
        n_epochs: int,
        learning_rate: Callable = fixed_learning_rate(0.1),
    ) -> Experiment:
        return Experiment(
            learning_strategy=learning_strategy,
            dataset=dataset,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )

    @staticmethod
    def create_random_experiment(
        n_points: int = 100,
        n_neurons: int = 2,
        dimension: int = 2,
        n_epochs: int = 2,
        initializer: Callable = InitializerFactory.zero_initializer,
    ) -> Experiment:
        learning_strategy = RandomStrategy(
            proximity_function=euclidean_distance,
            neural_network=initializer(n_neurons=n_neurons, n_dimensions=dimension),
        )
        dataset = Dataset.dummy(dimension=dimension, len=n_points)
        learning_rate = fixed_learning_rate(0.2)

        return ExperimentFactory.create_experiment(
            dataset=dataset,
            learning_strategy=learning_strategy,
            n_neurons=n_neurons,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )
