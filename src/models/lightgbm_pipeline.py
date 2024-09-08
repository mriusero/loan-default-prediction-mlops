from .skeleton import ModelPipeline

def run_lgb_pipeline(optimize, n_trials, exp_name):
    """
    Run a pipeline for LightGBM using ModelPipeline.

    :param optimize: Whether to optimize hyperparameters.
    :param n_trials: Number of trials for hyperparameter optimization if optimize is True.
    :param exp_name: Experiment name for MLflow.
    """
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_error',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42
    }

    pipeline = ModelPipeline(model_type='lightgbm', params=lgb_params, exp_name=exp_name,
                             save_path=f'./models/lightgbm.pkl')

    pipeline.run_pipeline(optimize=optimize, n_trials=n_trials)
