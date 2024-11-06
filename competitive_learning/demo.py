import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from competitive_learning.competitive_learning_app import CompetitiveLearningApp
from competitive_learning.enums import (
    available_initializers,
    available_learning_rate_functions,
    available_proximity_functions,
    available_strategies,
)


def load_json(file):
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return None


if "app_instance" not in st.session_state:
    st.session_state.app_instance = CompetitiveLearningApp()

app_instance = st.session_state.app_instance

# Define the Streamlit app
st.title("Competitive Learning")

### Load Dataset
st.header("Upload CSV")
csv_file = st.file_uploader("Upload CSV", type=["csv"])

if csv_file is not None:
    # Process the CSV file using the file path
    app_instance.process_csv(csv_file)

    st.write("### Dataset")
    st.dataframe(app_instance.dataframe)
    # st.json(app_instance.dataset.model_dump())
    FEATURES = {
        column: value
        for value, column in enumerate(app_instance.dataframe.columns.tolist())
    }

    ### Dataset Visualization
    st.header("Dataset Visualization")
    with st.form(key="dataviz"):
        st.write("### Scatter Plot")
        x_axis = st.selectbox("X-axis", FEATURES.keys())
        y_axis = st.selectbox("Y-axis", FEATURES.keys())
        color = st.selectbox("Color", FEATURES.keys())
        st.form_submit_button("Submit")

    if x_axis and y_axis:
        st.write("Scatter Plot")
        color = color if color else None
        st.scatter_chart(app_instance.dataframe, x=x_axis, y=y_axis, color=color)

    ### Experiment Setup
    st.header("Experiment Setup")

    previous_experiment = st.file_uploader("Upload Realization", type=["json"])
    if not previous_experiment:
        with st.form(key="experiment_setup"):
            epochs = st.number_input("Epochs", min_value=1, value=1)
            start_learning_rate = st.slider(
                "Start Learning Rate", 0.01, 1.0, 0.01, 0.001
            )
            learning_rate_function = st.selectbox(
                "Learning Rate Function", available_learning_rate_functions
            )
            epochs_for_learning_rate = st.number_input(
                "Epochs for Learning Rate", min_value=1, value=1
            )
            n_neurons = st.number_input("Number of Neurons", min_value=1, value=1)
            neurons_initializer = st.selectbox(
                "Neurons Initializer", available_initializers
            )
            strategy = st.selectbox("Strategy", available_strategies)
            proximity_function = st.selectbox(
                "Proximity Function", available_proximity_functions
            )
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            app_instance.setup_experiment(
                epochs,
                start_learning_rate,
                learning_rate_function,
                epochs_for_learning_rate,
                n_neurons,
                neurons_initializer,
                strategy,
                proximity_function,
            )
            app_instance.create_experiment()
            st.success("Experiment setup complete!")

    if previous_experiment:
        previous_experiment = load_json(previous_experiment)
        st.json(previous_experiment)
        with st.form(key="experiment_setup"):
            epochs = st.number_input("Epochs", min_value=1, value=1)
            start_learning_rate = st.slider(
                label="Start Learning Rate",
                min_value=0.01,
                max_value=1.0,
                value=previous_experiment["end_learning_rate"],
                step=0.001,
            )
            learning_rate_function = st.selectbox(
                "Learning Rate Function", available_learning_rate_functions
            )
            epochs_for_learning_rate = st.number_input(
                "Epochs for Learning Rate", min_value=1, value=1
            )
            strategy = st.selectbox("Strategy", available_strategies)
            proximity_function = st.selectbox(
                "Proximity Function", available_proximity_functions
            )
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            app_instance.create_experiment_from_previous(
                epochs,
                start_learning_rate,
                learning_rate_function,
                epochs_for_learning_rate,
                previous_experiment["neural_network"],
                strategy,
                proximity_function,
            )
            st.success("Experiment setup complete!")

    ### Run Experiment
    st.header("Run Experiment")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run Experiment"):
            app_instance.experiment.run_experiment()
    with col2:
        if st.button("Run Epoch"):
            app_instance.experiment.run_epoch()

    with col3:
        if st.button("Run Step"):
            app_instance.experiment.run_step()

    ### Experiment Visualization
    st.header("Experiment Visualization")

    steps, epoch, learning_rate = 0, 0, 0
    if app_instance.experiment is not None:
        col1, col2, col3 = st.columns(
            3,
        )
        steps = (
            app_instance.experiment.state_vector.n[-1] + 1
            if len(app_instance.experiment.state_vector.n) > 0
            else 0
        )
        epoch = (
            app_instance.experiment.state_vector.epoch[-1] + 1
            if len(app_instance.experiment.state_vector.epoch) > 0
            else 0
        )
        learning_rate = (
            app_instance.experiment.state_vector.learning_rate[-1]
            if len(app_instance.experiment.state_vector.learning_rate) > 0
            else 0
        )
        with col1:
            st.metric("Learning Steps", steps)
        with col2:
            st.metric("Epoch", epoch)
        with col3:
            st.metric("Learning Rate", learning_rate)

        if app_instance.experiment.step == app_instance.experiment.n_states:
            st.markdown("# :green[Experiment completed!]")

        if st.button("Reset Experiment"):
            app_instance.create_experiment()
            st.success("Experiment reset!")

        st.write("## Viz")
        col1, col2 = st.columns([0.8, 0.2])
        with col2:
            with st.form(key="viz"):
                x_axis = st.selectbox("X-axis", FEATURES.keys())
                y_axis = st.selectbox("Y-axis", FEATURES.keys())
                color = st.selectbox("Color", FEATURES.keys())
                st.form_submit_button("Submit")
        with col1:
            st.altair_chart(
                app_instance.experiment.plot_dataset_with_neurons(
                    app_instance.dataframe, x_axis, y_axis, FEATURES, color=color
                ),
                use_container_width=True,
            )

        # st.dataframe(app_instance.experiment.state_vector.df)
        # st.dataframe(app_instance.experiment.state_vector.get_quantization_error())
        st.altair_chart(
            app_instance.experiment.plot_learning_rate(),
            use_container_width=True,
        )
        st.altair_chart(
            app_instance.experiment.plot_quantization_error(),
            use_container_width=True,
        )

        if st.button("Save Experiment"):
            data = {
                # "id": app_instance.experiment.id,
                "strategy": app_instance.strategy,
                "end_learning_rate": app_instance.experiment.state_vector.learning_rate[
                    -1
                ],
                "learning_rate_function": app_instance.learning_rate_function,
                "epochs": app_instance.epochs,
                "n_neurons": app_instance.n_neurons,
                "neurons_initializer": app_instance.neurons_initializer,
                "proximity_function": app_instance.proximity_function,
                # "mean_quantization_error": app_instance.experiment.state_vector.df.quantization_error.mean(),
                "neural_network": app_instance.experiment.learning_strategy.neural_network.model_dump(),
            }

            file_path = f"output/experiment_{str(app_instance.experiment.id)}.json"

            # Save the data dictionary to a JSON file
            with open(file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)

    st.markdown("# Neurons Visualization")
    if st.button("Show Neurons Data"):
        st.dataframe(
            app_instance.experiment.learning_strategy.neural_network.get_neurons_report(
                app_instance.dataset
            )
        )
