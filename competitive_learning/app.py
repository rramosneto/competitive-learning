import os
import random
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import pandas as pd

from competitive_learning.competitive_learning_app import CompetitiveLearningApp

# Create an instance of the class
app_instance = CompetitiveLearningApp()

# Define the Gradio app
with gr.Blocks() as app:
    gr.Markdown("## Competitive Learning")

    ### Load Dataset
    gr.Markdown("### Upload CSV")
    with gr.Row():
        csv_input = gr.File(label="Upload CSV")
        load_button = gr.Button("Load CSV")

    ### Present Dataset
    gr.Markdown("### Dataset")
    dataframe_output = gr.Dataframe(scale=1, label="Dataframe")
    dataset_output = gr.JSON(scale=2, label="Dataset Info")
    x_axis = None
    y_axis = None
    color = None

    @gr.render(inputs=[dataframe_output])
    def update_scatter_plot(dataframe_output):
        if isinstance(app_instance.dataframe, pd.DataFrame):
            x_axis = gr.Radio(list(app_instance.dataframe.columns), label="X-axis")
            y_axis = gr.Radio(list(app_instance.dataframe.columns), label="Y-axis")
            color = gr.Textbox(label="Color")

    load_button.click(
        app_instance.process_csv,
        inputs=csv_input,
        outputs=[dataframe_output, dataset_output],
    )

    with gr.Row():
        gr.Markdown("### Dataset Visualization")
        if isinstance(app_instance.dataframe, pd.DataFrame):
            scatter_plot = gr.ScatterPlot(
                value=app_instance.dataframe,
                x=x_axis,
                y=y_axis,
                color=color,
                label="Scatter Plot",
            )

    with gr.Row():
        gr.Markdown("### Experiment Setup")
        start_learning_rate = gr.Slider(0.01, 1, 0.01, label="Start Learning Rate")
        learning_rate_function = gr.Dropdown(
            ["constant"], label="Learning Rate Function"
        )
        epochs = gr.Number(1, label="Epochs")
        n_neurons = gr.Number(1, label="Number of Neurons")
        neurons_initializer = gr.Dropdown(
            ["zero_initializer"], label="Neurons Initializer"
        )
        strategy = gr.Dropdown(["random"], label="Strategy")
        proximity_function = gr.Dropdown(
            ["euclidean_distance"], label="Proximity Function"
        )

        setup_button = gr.Button("Setup Experiment")

    setup_button.click(
        app_instance.setup_experiment,
        inputs=[
            start_learning_rate,
            learning_rate_function,
            epochs,
            n_neurons,
            neurons_initializer,
            strategy,
            proximity_function,
        ],
    )

    with gr.Row():
        gr.Markdown("### Experiment\n\n")

    with gr.Row():
        create_experiment_button = gr.Button("Create Experiment")
        output = gr.Textbox(label="Experiment Info")

    create_experiment_button.click(
        app_instance.create_experiment,
        inputs=None,
        outputs=output,
    )

# Launch the app
app.launch()
