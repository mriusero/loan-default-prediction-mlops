from .skeleton import ModelPipeline

def run_xgboost_pipeline(optimize, n_trials, exp_name):
    """
    Run a pipeline for XGBoost using ModelPipeline.

    :param optimize: Whether to optimize hyperparameters.
    :param n_trials: Number of trials for hyperparameter optimization if optimize is True.
    :param exp_name: Experiment name for MLflow.
    """
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'gamma': 0,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_state': 42
    }

    pipeline = ModelPipeline(model_type='xgboost', params=xgb_params, exp_name=exp_name,
                             save_path=f'./models/xgboost_model.pkl')

    pipeline.run_pipeline(optimize=optimize, n_trials=n_trials)