import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from competitive_learning.model import ExperimentFactory


def main():
    experiment = ExperimentFactory.create_random_experiment()

    experiment.run_experiment()

    print("Experiment finished")


if __name__ == "__main__":
    main()
