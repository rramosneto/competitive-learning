from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from typing_extensions import Annotated
from uuid import UUID, uuid4

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field, field_serializer, model_validator
from pydantic.config import ConfigDict  # noqa: F401
from pydantic.functional_serializers import PlainSerializer

from competitive_learning.function import euclidean_distance, fixed_learning_rate
from numpydantic import NDArray, Shape

ID = Annotated[UUID, PlainSerializer(lambda x: str(x), return_type=str)]
Array = Annotated[np.ndarray, PlainSerializer(lambda x: x.tolist(), return_type=list)]


class DataPoint(BaseModel):
    id: ID = Field(default_factory=uuid4)
    coordinates: (
        Array  # NDArray[Shape["*, ..."], np.float64]  # npt.NDArray[np.float64]
    )

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
    id: ID = Field(default_factory=uuid4)
    data_points: Tuple[DataPoint, ...]
    available_points: Set[int] = Field(default_factory=set)
    used_points: Set[int] = Field(default_factory=set)
    reset_count: int = 0

    def model_post_init(self, __context) -> None:
        self.available_points = set(range(len(self.data_points)))

    @property
    def mean_vector(self) -> float:
        return np.mean([dp.coordinates for dp in self.data_points], axis=0)

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
    def from_pandas(cls, df: pd.DataFrame) -> "Dataset":
        data_points = []
        for _, row in df.iterrows():
            data_points.append(DataPoint(coordinates=row.to_numpy()))
        return cls(data_points=tuple(data_points))

    @classmethod
    def dummy(cls, dimension, len) -> "Dataset":
        return cls(data_points=tuple(DataPoint.dummy(dimension) for _ in range(len)))


class Neuron(DataPoint):
    iterations: int = 0
    distances: Array = np.array([])  # Field(default_factory=np.array)
    relevance_score: float = 0.0

    def update_weights(self, coordinates: Array, distance: float) -> None:
        self.iterations += 1
        self.coordinates = coordinates
        self.distances = np.append(self.distances, distance)


class NeuralNetwork(BaseModel):
    id: ID = Field(default_factory=uuid4)
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

    @classmethod
    def dummy(cls, dimension: int) -> "NeuralNetwork":
        return cls(neurons=(Neuron.dummy(dimension), Neuron.dummy(dimension)))

    @classmethod
    def from_dict(cls, data):
        neurons = [
            Neuron(coordinates=np.array(neuron["coordinates"]))
            for neuron in data["neurons"]
        ]
        return cls(id=data["id"], neurons=tuple(neurons))

    def get_neurons_report(self, dataset: Dataset) -> pd.DataFrame:
        self.calculate_relevance(dataset)
        row_list = []
        for i, neuron in enumerate(self.neurons):
            row = {}
            row["neuron"] = i
            row["iterations"] = neuron.iterations
            row["travelled_distance"] = neuron.distances.sum()
            row["relevance"] = neuron.relevance_score
            row_list.append(row)

        df = pd.DataFrame(row_list).sort_values("relevance", ascending=False)

        return df

    def assign_data_points(self, dataset: Dataset) -> Dict[int, List[DataPoint]]:
        assignments = {i: [] for i in range(len(self.neurons))}

        for dp in dataset.data_points:
            closest_neuron_index = np.argmin(
                [
                    np.linalg.norm(dp.coordinates - neuron.coordinates)
                    for neuron in self.neurons
                ]
            )
            assignments[closest_neuron_index].append(dp)

        return assignments

    def calculate_relevance(self, dataset: Dataset) -> None:
        assignments = self.assign_data_points(dataset)

        for i, neuron in enumerate(self.neurons):
            neuron.relevance_score = len(assignments[i]) / dataset.len


class LearningStrategy(BaseModel, ABC):
    id: ID = Field(default_factory=uuid4)
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


class WTAStrategy(LearningStrategy):
    """
    Winner Takes All strategy for learning. The neuron closest to the data point is chosen.
    """

    def choose_neurons(self, data_point: DataPoint) -> Tuple[Tuple[Neuron, ...], float]:
        min_distance = float("inf")
        closest_neuron = None

        for neuron in self.neural_network.neurons:
            distance = self.proximity_function(
                data_point.coordinates, neuron.coordinates
            )
            if distance < min_distance:
                min_distance = distance
                closest_neuron = neuron

        return (closest_neuron,), min_distance

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
            distance = self.proximity_function(data_point.coordinates, coordinates)
            neuron.update_weights(coordinates=coordinates, distance=distance)

    @classmethod
    def dummy(cls) -> "RandomStrategy":
        return cls(
            proximity_function=euclidean_distance,
            learning_rate=fixed_learning_rate(0.2),
        )


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
            distance = self.proximity_function(data_point.coordinates, coordinates)
            neuron.update_weights(coordinates=coordinates, distance=distance)

    @classmethod
    def dummy(cls) -> "RandomStrategy":
        return cls(
            proximity_function=euclidean_distance,
            learning_rate=fixed_learning_rate(0.2),
        )


class StateVector(BaseModel):
    n: List[int] = Field(default_factory=list)
    epoch: List[int] = Field(default_factory=list)
    iteration: List[int] = Field(default_factory=list)
    active_data_point: List[ID] = Field(default_factory=list)
    active_neurons: List[List[ID]] = Field(default_factory=list)
    learning_rate: List[float] = Field(default_factory=list)
    quantization_error: List[float] = Field(default_factory=list)
    df: Optional[pd.DataFrame] = Field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "n",
                "epoch",
                "iteration",
                "active_data_point",
                "active_neurons",
                "learning_rate",
                "quantization_error",
            ]
        )
    )

    def update(
        self,
        step: int,
        epoch: int,
        iteration: int,
        active_data_point: ID,
        active_neurons: List[ID],
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

        # Update the DataFrame
        new_row = {
            "n": step,
            "epoch": epoch,
            "iteration": iteration,
            "active_data_point": active_data_point,
            "active_neurons": active_neurons,
            "learning_rate": learning_rate,
            "quantization_error": quantization_error,
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def get_quantization_error(self) -> pd.DataFrame:
        def mean_squared(s: pd.Series):
            return np.mean(s**2)

        grouped = (
            self.df.groupby("epoch")["quantization_error"]
            .agg(mean_squared)
            .to_frame()
            .reset_index()
        )

        return grouped

    class Config:
        arbitrary_types_allowed = True


class LearningRate(BaseModel):
    def __init__(
        self, learning_rate_type: str, initial_learning_rate: float, n_states: int
    ) -> None:
        self.learning_rate_type = learning_rate_type
        self.initial_learning_rate = initial_learning_rate


class Experiment(BaseModel):
    id: ID = Field(default_factory=uuid4)
    learning_strategy: LearningStrategy
    learning_rate: Callable
    dataset: Dataset
    n_epochs: int
    step: int = 0
    state: int = 0
    epoch: int = 0
    state_vector: StateVector = Field(default_factory=StateVector)

    def reset(self) -> None:
        self.step = 0
        self.state = 0
        self.epoch = 0
        self.state_vector = StateVector()

    def run_experiment(self) -> None:
        while self.epoch < self.n_epochs:
            self.run_epoch()
        return

    def run_to_end(self) -> None:
        while self.step < self.n_states:
            self.run_step()

    def run_epoch(self) -> None:
        if self.epoch < self.n_epochs:
            while self.state < self.dataset.len - 1:
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
            active_data_point=str(data_point.id),
            active_neurons=[str(n.id) for n in neurons],
            learning_rate=learning_rate,
            quantization_error=distance,
        )
        self.step += 1
        self.state += 1

    @property
    def n_states(self) -> int:
        return self.n_epochs * self.dataset.len

    def plot_quantization_error(self):
        chart = (
            alt.Chart(self.state_vector.get_quantization_error())
            .mark_line()
            .encode(x="epoch", y="quantization_error")
            .properties(title="Quantization Error")
        )
        return chart

    def plot_learning_rate(self):
        chart = (
            alt.Chart(self.state_vector.df)
            .mark_line()
            .encode(x="n", y="learning_rate")
            .properties(title="Learning Rate")
        )
        return chart

    def plot_dataset_with_neurons(
        self,
        data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        features_dict: Dict[str, int],
        color=None,
    ):
        # Create the base chart for the dataset
        chart = (
            alt.Chart(data)
            .mark_circle()
            .encode(x=x_axis, y=y_axis, color=color if color else alt.value("blue"))
            .properties(title="Dataset with Neurons")
        )

        # Create a DataFrame for the neurons
        neurons_df = pd.DataFrame(
            {
                "x": [
                    n.coordinates[features_dict[x_axis]]
                    for n in self.learning_strategy.neural_network.neurons
                ],
                "y": [
                    n.coordinates[features_dict[y_axis]]
                    for n in self.learning_strategy.neural_network.neurons
                ],
            }
        )

        # Create the chart for the neurons
        neurons_chart = (
            alt.Chart(neurons_df).mark_point(color="red", size=100).encode(x="x", y="y")
        )

        # Combine the dataset chart and the neurons chart
        combined_chart = chart + neurons_chart

        return combined_chart


class InitializerFactory(BaseModel):
    @staticmethod
    def zero_initializer(
        n_neurons: int, n_dimensions: int, dataset: Dataset
    ) -> NeuralNetwork:
        return NeuralNetwork(
            neurons=tuple(
                [Neuron(coordinates=np.zeros(n_dimensions)) for _ in range(n_neurons)]
            )
        )

    @staticmethod
    def mean_initializer(
        n_neurons: int, n_dimensions: int, dataset: Dataset
    ) -> NeuralNetwork:
        return NeuralNetwork(
            neurons=tuple(
                [
                    Neuron(coordinates=np.array(dataset.mean_vector))
                    for _ in range(n_neurons)
                ]
            )
        )


class ExperimentFactory(BaseModel):
    @staticmethod
    def create_experiment(
        dataset: Dataset,
        learning_strategy: LearningStrategy,
        n_epochs: int,
        learning_rate: Callable,
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
        dataset = Dataset.dummy(dimension=dimension, len=n_points)
        learning_rate = fixed_learning_rate(0.2)
        learning_strategy = RandomStrategy(
            proximity_function=euclidean_distance,
            neural_network=initializer(
                n_neurons=n_neurons, n_dimensions=dimension, dataset=dataset
            ),
        )

        return ExperimentFactory.create_experiment(
            dataset=dataset,
            learning_strategy=learning_strategy,
            n_neurons=n_neurons,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )
