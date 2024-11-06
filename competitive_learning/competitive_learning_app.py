from competitive_learning.enums import (
    INITIALIZER,
    LEARNING_RATE_FUNCTION,
    PROXIMITY_FUNCTION,
    STRATEGY,
)
from competitive_learning.model import (
    Experiment,
    ExperimentFactory,
    LearningStrategy,
    NeuralNetwork,
)
from competitive_learning.handler.data import load_csv


class CompetitiveLearningApp:
    def __init__(self):
        self.strategy = None
        self.dataframe = None
        self.dataset = None
        self.experiment = None
        self.epochs = None
        self.start_learning_rate = None
        self.learning_rate_function = None
        self.epochs_for_learning_rate = None
        self.proximity_function = None
        self.n_neurons = None
        self.neurons_initializer = None

    def process_csv(self, file):
        self.dataframe, self.dataset = load_csv(file)

    def setup_experiment(
        self,
        epochs,
        start_learning_rate,
        learning_rate_function,
        epochs_for_learning_rate,
        n_neurons,
        neurons_initializer,
        strategy,
        proximity_function,
    ):
        self.epochs = epochs
        self.start_learning_rate = start_learning_rate
        self.learning_rate_function = learning_rate_function
        self.epochs_for_learning_rate = epochs_for_learning_rate
        self.n_neurons = n_neurons
        self.neurons_initializer = neurons_initializer
        self.strategy = strategy
        self.proximity_function = proximity_function

    def create_experiment_from_previous(
        self,
        epochs,
        start_learning_rate,
        learning_rate_function,
        epochs_for_learning_rate,
        neural_network,
        strategy,
        proximity_function,
    ):
        self.epochs = epochs
        self.start_learning_rate = start_learning_rate
        self.learning_rate_function = learning_rate_function
        self.epochs_for_learning_rate = epochs_for_learning_rate
        self.strategy = strategy
        self.proximity_function = proximity_function

        learning_rate = self.create_learning_rate()
        strategy = STRATEGY[self.strategy](
            proximity_function=self.create_proximity_function(),
            neural_network=NeuralNetwork.from_dict(neural_network),
        )
        self.experiment = ExperimentFactory.create_experiment(
            learning_strategy=strategy,
            dataset=self.dataset,
            n_epochs=self.epochs,
            learning_rate=learning_rate,
        )

    def create_experiment(self):
        learning_rate = self.create_learning_rate()
        strategy = self.create_strategy()
        self.experiment = ExperimentFactory.create_experiment(
            learning_strategy=strategy,
            dataset=self.dataset,
            n_epochs=self.epochs,
            learning_rate=learning_rate,
        )

    def create_strategy(self):
        return STRATEGY[self.strategy](
            proximity_function=self.create_proximity_function(),
            neural_network=self.create_neural_network(),
        )

    def create_learning_rate(self):
        return LEARNING_RATE_FUNCTION[self.learning_rate_function](
            learning_rate=self.start_learning_rate,
            n_states=self.epochs_for_learning_rate * self.dataset.len,
            # n_states=self.epochs * self.dataset.len,
        )

    def create_proximity_function(self):
        return PROXIMITY_FUNCTION[self.proximity_function]

    def create_neural_network(self):
        initializer = INITIALIZER[self.neurons_initializer]
        neural_network = initializer(
            self.n_neurons, self.dataset.dimension, self.dataset
        )
        return neural_network

    def load_experiment(self, experiment):
        self.experiment = experiment
