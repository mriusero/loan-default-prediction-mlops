import streamlit as st
import pickle
import subprocess
import os
import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tempfile

def configure_mlflow(experiment_name="Default"):
    """
    Configure MLflow to use a specific experiment.

    :param experiment_name: Name of the MLflow experiment.
    """
    mlflow.set_experiment(experiment_name)

class ModelPipeline:
    def __init__(self, model_type, params, exp_name="Default", save_path=None):
        """
        Initialize the model pipeline.

        :param model_type: Type of the model ('random_forest' or 'logistic_regression').
        :param params: Dictionary of hyperparameters for the model.
        :param experiment_name: Name of the MLflow experiment.
        :param save_path: Path to save the trained model locally if not using MLflow.
        """
        self.model_type = model_type
        self.params = params
        self.experiment_name = exp_name
        self.save_path = save_path
        self.model = None

        # Configure MLflow for the experiment
        configure_mlflow(exp_name)

    def initialize_model(self, params=None):
        """
        Instantiate the model based on the specified type and parameters.

        :param params: Optional dictionary of hyperparameters.
        """
        params = params or self.params

        # Initialize the model based on the specified type
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**params)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**params)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

    def train(self, X_train, y_train):
        """
        Train the model using the training data.

        :param X_train: Feature matrix for training.
        :param y_train: Target vector for training.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call 'initialize_model()' first.")
        self.model.fit(X_train, y_train)

    def validate(self, X_test, y_test):
        """
        Validate the model and return accuracy.

        :param X_test: Feature matrix for testing.
        :param y_test: Target vector for testing.
        :return: Accuracy of the model.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call 'initialize_model()' first.")
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def log_params(self):
        """
        Log model parameters to MLflow.
        """
        mlflow.log_params(self.params)
        print(f"- parameters logged to MLflow: {self.params}")

    def log_metrics(self, accuracy):
        """
        Log model metrics to MLflow.

        :param accuracy: Accuracy score to log.
        """
        mlflow.log_metric("accuracy", accuracy)
        print(f"-- accuracy logged to MLflow: {accuracy}")

    def log_artifacts(self):
        """
        Save training logs and plots to MLflow.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save training log
            log_path = os.path.join(tmpdirname, "training_log.txt")
            with open(log_path, 'w') as f:
                f.write("Training completed successfully.")
            mlflow.log_artifact(log_path)
            print(f"--- training log saved and logged to MLflow.")

            # Save example plots
            plt.figure()
            plt.title("Confusion Matrix")
            plot_path = os.path.join(tmpdirname, "confusion_matrix.png")
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            mlflow.autolog()
            print(f"---- plot saved and logged to MLflow.")

    def log_model(self, X_sample):
        """
        Log the trained model to MLflow and/or save locally with versioning.

        :param X_sample: A small sample of the input data to infer the model's input signature.
        :return: URI of the logged model if logged to MLflow.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Call 'initialize_model()' first.")

        # Save the model locally if a path is provided
        if self.save_path:
            new_file_path = self._get_versioned_file_path()
            with open(new_file_path, 'wb') as f:                           #Save locally the model
                pickle.dump(self.model, f)
            print(f"----- model saved locally at: {new_file_path}")

            mlflow.log_artifact(new_file_path)
            print(f"------ local model file '{new_file_path}' logged to MLflow as an artifact.")

    def _get_versioned_file_path(self):
        """
        Generate a new file path for the model with an incremented version number.

        :return: New file path with versioning.
        """
        directory = os.path.dirname(self.save_path)
        base_name = os.path.basename(self.save_path)
        base_name_no_ext, ext = os.path.splitext(base_name)

        # Get existing model files and extract version numbers
        existing_files = [f for f in os.listdir(directory) if f.startswith(base_name_no_ext) and f.endswith(ext)]
        version_numbers = [int(f[len(base_name_no_ext) + 1:-len(ext)]) for f in existing_files if
                           f[len(base_name_no_ext) + 1:-len(ext)].isdigit()]

        # Start versioning at 01, not 00
        next_version = max(version_numbers, default=0) + 1

        # Generate the new file path with zero-padded version number
        new_file_path = os.path.join(directory, f"{base_name_no_ext}_{next_version:02d}{ext}")
        return new_file_path

    def objective(self, trial, X_train, y_train, X_test, y_test):
        """
        Objective function for Optuna to optimize hyperparameters.

        :param trial: Optuna trial object.
        :param X_train: Feature matrix for training.
        :param y_train: Target vector for training.
        :param X_test: Feature matrix for testing.
        :param y_test: Target vector for testing.
        :return: Loss value to minimize.
        """
        # Define hyperparameter search space based on model type
        if self.model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': 42
            }
        elif self.model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_loguniform('C', 1e-3, 1e2),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported for optimization.")

        self.initialize_model(params)
        self.train(X_train, y_train)
        accuracy = self.validate(X_test, y_test)

        return 1 - accuracy  # Minimize 1 - accuracy to maximize accuracy

    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test, n_trials=10):
        """
        Optimize hyperparameters using Optuna.

        :param X_train: Feature matrix for training.
        :param y_train: Target vector for training.
        :param X_test: Feature matrix for testing.
        :param y_test: Target vector for testing.
        :param n_trials: Number of trials for Optuna optimization.
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
        self.params = study.best_params
        st.write("Best hyperparameters found: ", self.params)

    def run_pipeline(self, X, y, optimize=False, n_trials=10):
        """
        Run the complete ML pipeline.

        :param X: Feature matrix.
        :param y: Target vector.
        :param optimize: Whether to optimize hyperparameters.
        :param n_trials: Number of trials for hyperparameter optimization.
        """
        print(f"\n-------- Machine Learning Experience pipeline ---------")
        print(f"Experience: '{self.experiment_name}'")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Optimize hyperparameters before starting a new MLflow run
        if optimize:
            self.optimize_hyperparameters(X_train, y_train, X_test, y_test, n_trials)

        # Start a new MLflow run after optimization
        with mlflow.start_run() as run:
            self.initialize_model()  # Use the best parameters found if optimized
            self.train(X_train, y_train)
            accuracy = self.validate(X_test, y_test)

            # Log parameters and metrics after optimization
            self.log_params()
            self.log_metrics(accuracy)
            self.log_artifacts()
            self.log_model(X_train)

            # Add custom tags to the MLflow run
            mlflow.set_tag("model_version", f'{self.model_type}_{self.get_next_version()}')
            print(f"------- tags added to MLflow run: 'model_version', f'{self.model_type}_{self.get_next_version()}'")
            print("Experience completed successfully!")

    def get_next_version(self):
        """
        Get the next version number for the model.

        :return: Version number as a string, zero-padded.
        """
        directory = os.path.dirname(self.save_path)
        base_name = os.path.basename(self.save_path)
        base_name_no_ext, ext = os.path.splitext(base_name)

        # Get existing model files and extract version numbers
        existing_files = [f for f in os.listdir(directory) if f.startswith(base_name_no_ext) and f.endswith(ext)]
        version_numbers = [int(f[len(base_name_no_ext) + 1:-len(ext)]) for f in existing_files if
                           f[len(base_name_no_ext) + 1:-len(ext)].isdigit()]

        # Start versioning at 01, not 00
        next_version = max(version_numbers, default=0)
        return f"{next_version:03d}"  # Ensure version number is zero-padded to 3 digits

    def predict(self, X):
        """
        Predict using the trained model.

        :param X: Feature matrix for prediction.
        :return: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model is not initialized or trained.")
        return self.model.predict(X)

    @staticmethod
    def load_model(model_uri):
        """
        Load a model from MLflow.

        :param model_uri: URI of the model stored in MLflow.
        :return: Loaded model.
        """
        return mlflow.sklearn.load_model(model_uri)


