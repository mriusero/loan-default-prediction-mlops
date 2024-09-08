# logr_pipeline.py

from .skeleton import ModelPipeline

def run_logr_pipeline(optimize, n_trials, exp_name):
    """
    Run a pipeline for LogisticRegression using ModelPipeline.

    :param model_name:
    :param exp_name:
    :param optimize: Whether to optimize hyperparameters.
    :param n_trials: Number of trials for hyperparameter optimization if optimize is True.
    """
    lr_params = {
        'C': 1.0,
        'max_iter': 200,
        'random_state': 42
    }

    pipeline = ModelPipeline(model_type='logistic_regression', params=lr_params, exp_name=exp_name,
                             save_path=f'./models/logistic_regression.pkl')

    pipeline.run_pipeline(optimize=optimize, n_trials=n_trials)