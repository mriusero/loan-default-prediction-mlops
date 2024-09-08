# logr_pipeline.py

import streamlit as st
from sklearn.datasets import load_iris

from .skeleton import ModelPipeline

def run_logr_pipeline(optimize, n_trials, exp_name):
    """
    Run a pipeline for LogisticRegression using ModelPipeline.

    :param model_name:
    :param exp_name:
    :param optimize: Whether to optimize hyperparameters.
    :param n_trials: Number of trials for hyperparameter optimization if optimize is True.
    """
    # For Template
    data = load_iris()
    X, y = data.data, data.target

    # Default
    lr_params = {
        'C': 1.0,
        'max_iter': 200,
        'random_state': 42
    }

    # Instance model
    pipeline = ModelPipeline(model_type='logistic_regression', params=lr_params, exp_name=exp_name,
                             save_path=f'./models/logistic_regression.pkl')

    # Execute pipeline (with or without hyperparameters optimization)
    pipeline.run_pipeline(X, y, optimize=optimize, n_trials=n_trials)

    X_new = [[5.1, 3.5, 1.4, 0.2]]
    predictions = pipeline.predict(X_new)
    st.write("Prédictions avec LogisticRegression :", predictions)

    pipeline.start_mlflow_ui()


