This project explores the use of AutoML frameworks to automate the process of detecting fraudulent transactions in credit card datasets. By leveraging libraries such as H2O AutoML and FLAML, the project demonstrates how to train, evaluate, and deploy machine learning models for fraud detection with minimal manual intervention.

Features

Data Preprocessing:

Reads and preprocesses the credit card fraud dataset.

Scales features using StandardScaler for model compatibility.

Splits the dataset into training and testing sets.

Model Training:

Utilizes H2O AutoML to automatically select the best model from a range of algorithms.

Implements FLAML for efficient and fast AutoML.

Compares multiple models using performance metrics.

Evaluation Metrics:

Accuracy, precision, recall, F1 score, and ROC-AUC.

Confusion matrix visualization with Seaborn heatmaps.

Deployment:

Integrates a simple user interface for fraud prediction using Gradio.

Dataset

Name: Credit Card Fraud Detection Dataset

Source: The dataset is loaded from a specified path. It contains anonymized credit card transactions labeled as fraudulent or normal.

Structure: Features include transaction attributes and a Class column for labels (1 for fraud and 0 for normal).

Project Workflow

Data Loading and Preprocessing

Load the dataset using Pandas and H2O.

Encode and scale features.

Split data into training and testing subsets.

AutoML Implementation

H2O AutoML:

Automatically trains multiple models and selects the best-performing one.

Provides a leaderboard of model performances.

FLAML:

Fast and lightweight AutoML for quick experimentation.

Model Evaluation

Use the test dataset to evaluate model accuracy, precision, recall, and F1 score.

Generate confusion matrices to visualize prediction performance.

Interactive Deployment

Build a Gradio-based web interface for predicting fraud based on transaction details.

Prerequisites

Python 3.7+

Required Libraries:

pandas
numpy
scikit-learn
seaborn
h2o
flaml
gradio

How to Run the Project

Clone the repository or download the project files.

Install the required libraries using:

pip install -r requirements.txt

Run the Jupyter Notebook (Automl.ipynb) to:

Load and preprocess the dataset.

Train models using AutoML frameworks.

Evaluate model performance.

Launch the Gradio interface for fraud prediction:

gr.Interface(fn=defection, inputs="number", outputs="text", title=headline).launch()

Key Results

Achieved significant model performance improvement using AutoML frameworks.

Built an intuitive user interface for real-time fraud detection.

Future Scope

Experiment with additional AutoML frameworks like AutoKeras or TPOT.

Integrate the model into a production-grade API or dashboard.

Explore advanced interpretability techniques to explain fraud predictions.

Acknowledgments

Thanks to H2O.ai for their robust AutoML tools.

Gradio for providing a simple and efficient UI framework.

