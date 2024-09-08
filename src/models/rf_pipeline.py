# rf_pipeline.py

import streamlit as st
from sklearn.datasets import load_iris

from .skeleton import ModelPipeline


def run_rf_pipeline(optimize, n_trials, exp_name):
    """
    Run a pipeline for RandomForestClassifier using ModelPipeline.

    :param optimize: Whether to optimize hyperparameters.
    :param n_trials: Number of trials for hyperparameter optimization if optimize is True.
    :param exp_name: Name of the MLflow experiment to use.
    """
    # Load example data (Iris dataset)
    data = load_iris()
    X, y = data.data, data.target

    # Define default hyperparameters for the RandomForestClassifier model
    rf_params = {
        'n_estimators': 100,  # Number of trees in the forest
        'max_depth': 5,       # Maximum depth of the trees
        'random_state': 42    # Random seed for reproducibility
    }

    # Create an instance of ModelPipeline for the RandomForestClassifier model
    pipeline = ModelPipeline(
        model_type='random_forest',   # Specify the model type
        params=rf_params,             # Pass the default hyperparameters
        exp_name=exp_name,            # Experiment name for MLflow
        save_path='./models/random_forest.pkl'  # Path to save the trained model
    )

    # Run the pipeline with or without hyperparameter optimization
    pipeline.run_pipeline(X, y, optimize=optimize, n_trials=n_trials)

    # Make predictions with the trained model
    X_new = [[5.1, 3.5, 1.4, 0.2]]  # Example of new data point for prediction
    predictions = pipeline.predict(X_new)
    st.write("Predictions with RandomForestClassifier:", predictions)

    # Start the MLflow UI to visualize experiments
    pipeline.start_mlflow_ui()
