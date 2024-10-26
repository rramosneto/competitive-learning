import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gradio as gr
import matplotlib.pyplot as plt

from competitive_learning.model import Dataset

# Initialize a variable to store the DataFrame
dataset = None
experiment = None


def load_dataset(file):
    global dataset
    dataset = Dataset.from_csv(file.name)
    return f"The dataset has {dataset.len} rows"


with gr.Blocks() as app:
    gr.Markdown("## Competitive Learning")

    with gr.Row():
        csv_input = gr.File(label="Upload CSV")
        load_button = gr.Button("Load CSV")

    dataset_preview = gr.Textbox(label="Dataset Preview")
    create_experiment_button = gr.Button("Plot Data")
    plot_output = gr.Image(label="Plot Output")

    load_button.click(load_csv, inputs=csv_input, outputs=output_df)
    plot_button.click(plot_data, outputs=plot_output)

# Launch the app
app.launch()
