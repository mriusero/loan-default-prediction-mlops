import os
import pickle
import tempfile
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score
)
from xgboost import XGBClassifier


def configure_mlflow(experiment_name="Default"):
    """Configure MLflow to use a specific experiment."""
    mlflow.set_experiment(experiment_name)

class ModelPipeline:
    def __init__(self, model_type, params, exp_name="Default", save_path=None):
        """
        Initialize the model pipeline.

        :param model_type: Type of the model ('random_forest', 'logistic_regression', 'xgboost', 'lightgbm').
        :param params: Dictionary of hyperparameters for the model.
        :param exp_name: Name of the MLflow experiment.
        :param save_path: Path to save the trained model locally.
        """
        self.model_type = model_type
        self.params = params
        self.experiment_name = exp_name
        self.save_path = save_path
        self.model = None
        configure_mlflow(exp_name)  # Configure MLflow for the experiment

    def initialize_model(self, params=None):
        """Instantiate the model based on the specified type and parameters."""
        params = params or self.params
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**params)
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(**params)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**params)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

        if not (hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict'))):
            raise ValueError("Model does not have a 'predict' method.")

    def train(self, X_train, y_train):
        """Train the model using the training data."""
        if self.model is None:
            raise ValueError("Model is not initialized. Call 'initialize_model()' first.")
        self.model.fit(X_train, y_train)

    def validate(self, X_val, y_val):
        """Validate the model and return various metrics."""
        if self.model is None:
            raise ValueError("Model is not initialized. Call 'initialize_model()' first.")
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'auc_roc': roc_auc_score(y_val, y_pred_proba),
            'pr_auc': average_precision_score(y_val, y_pred_proba),
            'f1': f1_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred)
        }
        return metrics

    def log_params(self):
        """Log model parameters to MLflow."""
        mlflow.log_params(self.params)
        print(f"- Parameters logged to MLflow: {self.params}")

    def log_metrics(self, metrics):
        """Log model metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            print(f"-- {key} logged to MLflow: {value}")

    def log_artifacts(self):
        """Save training logs and plots to MLflow."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            log_path = os.path.join(tmpdirname, "training_log.txt")
            with open(log_path, 'w') as f:
                f.write("Training completed successfully.")
            mlflow.log_artifact(log_path)
            print("--- Training log saved and logged to MLflow.")

            plt.figure()
            plt.title("Confusion Matrix")
            plot_path = os.path.join(tmpdirname, "confusion_matrix.png")
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            mlflow.autolog()
            print("---- Plot saved and logged to MLflow.")

    def log_model(self, X_sample):
        """Log the trained model to MLflow and/or save locally with versioning."""
        if self.model is None:
            raise ValueError("Model is not initialized. Call 'initialize_model()' first.")
        if self.save_path:
            new_file_path = self._get_versioned_file_path()
            with open(new_file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"----- Model saved locally at: {new_file_path}")

            signature = mlflow.models.infer_signature(X_sample, self.model.predict(X_sample))

            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                signature=signature
            )
            mlflow.log_artifact(new_file_path)
            print(f"------ Local model file '{new_file_path}' logged to MLflow as an artifact.")

    def _get_versioned_file_path(self):
        """Generate a new file path for the model with an incremented version number."""
        directory, base_name = os.path.dirname(self.save_path), os.path.basename(self.save_path)
        base_name_no_ext, ext = os.path.splitext(base_name)
        existing_files = [f for f in os.listdir(directory) if f.startswith(base_name_no_ext) and f.endswith(ext)]
        version_numbers = [int(f[len(base_name_no_ext) + 1:-len(ext)]) for f in existing_files if f[len(base_name_no_ext) + 1:-len(ext)].isdigit()]
        next_version = max(version_numbers, default=0) + 1
        return os.path.join(directory, f"{base_name_no_ext}_{next_version:02d}{ext}")

    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for Optuna to optimize hyperparameters.

        :param trial: Optuna trial object.
        :param X_train: Feature matrix for training.
        :param y_train: Target vector for training.
        :param X_val: Feature matrix for validation.
        :param y_val: Target vector for validation.
        :return: Loss value to minimize.
        """
        if self.model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': 42
            }
        elif self.model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }
        elif self.model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42
            }
        elif self.model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', -1, 50)
            }
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported for optimization.")

        self.initialize_model(params)
        self.train(X_train, y_train)

        y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_val_pred = self.model.predict(X_val)

        auc_roc = roc_auc_score(y_val, y_val_pred_proba)
        pr_auc = average_precision_score(y_val, y_val_pred_proba)
        f1 = f1_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)

        composite_score = (0.4 * auc_roc) + (0.3 * pr_auc) + (0.2 * f1) + (0.05 * recall) + (0.05 * precision)

        return -composite_score

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=10):
        """Optimize hyperparameters using Optuna."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)

            mlflow.autolog(disable=True)

            with mlflow.start_run(nested=True):
                study = optuna.create_study(direction='minimize')
                study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
                self.params = study.best_params
                st.write("Best hyperparameters found: ", self.params)

    def run_pipeline(self, X_selected, optimize=False, n_trials=10):
        """Run the complete ML pipeline."""
        print(f"\n-------- Machine Learning Experience Pipeline ---------")
        print(f"Experiment: '{self.experiment_name}'")

        from src.app import get_data_splits
        X_train, y_train, X_val, y_val, X_test, y_test = get_data_splits()

        X_train = X_train[X_selected]
        X_val = X_val[X_selected]
        X_test = X_test[X_selected]

        if optimize:
            self.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials)

        self.initialize_model()  # Use best params if optimized
        self.train(X_train, y_train)

        mlflow.autolog(log_input_examples=True, log_model_signatures=True)

        with mlflow.start_run() as run:
            metrics = self.validate(X_val, y_val)
            self.log_metrics(metrics)

            mlflow.set_tag("model_version", f'{self.model_type}_{self.get_next_version()}')
            mlflow.set_tag("experiment_name", self.experiment_name)
            print(f"------- Tags added to MLflow run: 'model_version', '{self.model_type}_{self.get_next_version()}'")
            print("Experience completed successfully!")

            self.display_results_on_streamlit(metrics, X_val, y_val)
            self.log_model(X_val)  # Pass a sample for signature inference

        predictions = self.predict(X_test)

    def get_next_version(self):
        """Get the next version number for the model."""
        directory, base_name = os.path.dirname(self.save_path), os.path.basename(self.save_path)
        base_name_no_ext, ext = os.path.splitext(base_name)
        existing_files = [f for f in os.listdir(directory) if f.startswith(base_name_no_ext) and f.endswith(ext)]
        version_numbers = [int(f[len(base_name_no_ext) + 1:-len(ext)]) for f in existing_files if f[len(base_name_no_ext) + 1:-len(ext)].isdigit()]
        next_version = max(version_numbers, default=0)
        return f"{next_version:03d}"  # Zero-padded to 3 digits

    def predict(self, X):
        """Predict using the trained model and return a DataFrame with customer_id and predictions."""
        if self.model is None:
            raise ValueError("Model is not initialized or trained.")

        predictions = self.model.predict(X)

        return predictions

    @staticmethod
    def load_model(model_uri):
        """Load a model from MLflow."""
        return mlflow.sklearn.load_model(model_uri)

    def display_results_on_streamlit(self, metrics, X_val, y_val):
        """Display the model's parameters, accuracy, and confusion matrix on Streamlit."""
        st.write(f"## Experiment: {self.experiment_name}")
        st.write(f"### Model Type: {self.model_type}")
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            st.write("### Parameters:")
            st.json(self.params)
        with col2:
            st.write("### Metrics:")
            for key, value in metrics.items():
                st.write(f"{key}: {value:.4f}")
        with col3:
            st.write("### Confusion Matrix:")
            y_pred = self.model.predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)

        mlflow.end_run()
