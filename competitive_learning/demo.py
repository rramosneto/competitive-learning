import os
import random
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from competitive_learning.competitive_learning_app import CompetitiveLearningApp

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

    ### Dataset Visualization
    st.header("Dataset Visualization")
    with st.form(key="dataviz"):
        st.write("### Scatter Plot")
        x_axis = st.selectbox("X-axis", app_instance.dataframe.columns)
        y_axis = st.selectbox("Y-axis", app_instance.dataframe.columns)
        color = st.selectbox("Color", app_instance.dataframe.columns)
        st.form_submit_button("Submit")

    if x_axis and y_axis:
        st.write("Scatter Plot")
        color = color if color else None
        st.scatter_chart(app_instance.dataframe, x=x_axis, y=y_axis, color=color)

    ### Experiment Setup
    st.header("Experiment Setup")
    with st.form(key="experiment_setup"):
        start_learning_rate = st.slider("Start Learning Rate", 0.01, 1.0, 0.01)
        learning_rate_function = st.selectbox("Learning Rate Function", ["constant"])
        epochs = st.number_input("Epochs", min_value=1, value=1)
        n_neurons = st.number_input("Number of Neurons", min_value=1, value=1)
        neurons_initializer = st.selectbox(
            "Neurons Initializer", ["zero_initializer", "mean_initializer"]
        )
        strategy = st.selectbox("Strategy", ["random", "wta"])
        proximity_function = st.selectbox("Proximity Function", ["euclidean_distance"])
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        app_instance.setup_experiment(
            start_learning_rate,
            learning_rate_function,
            epochs,
            n_neurons,
            neurons_initializer,
            strategy,
            proximity_function,
        )
        st.success("Experiment setup complete!")

    ### Create Experiment
    st.header("Experiment")
    if st.button("Create Experiment"):
        app_instance.create_experiment()
        st.success("Experiment created!")
    st.metric("My metric", 42, 2)

    ### Run Experiment
    st.header("Run Experiment")
    col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
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

    col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
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
    st.altair_chart(
        app_instance.experiment.plot_dataset_with_neurons(
            app_instance.dataframe, x=x_axis, y=y_axis, color=color
        ),
        use_container_width=True,
    )
    st.altair_chart(
        app_instance.experiment.plot_quantization_error(),
        use_container_width=True,
    )

    st.altair_chart(
        app_instance.experiment.plot_learning_rate(),
        use_container_width=True,
    )
