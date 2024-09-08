import os
import yaml
import json
import subprocess
import webbrowser

def load_yaml(file_path):
    """Load a YAML file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def check_mlruns_directory(mlruns_path):
    """Check if the 'mlruns' directory exists."""
    if not os.path.exists(mlruns_path):
        raise FileNotFoundError(f"The specified directory '{mlruns_path}' does not exist.")

def get_mlruns_data(mlruns_path):
    """Retrieve information about experiments, runs, artifacts, metrics, params, and tags in the 'mlruns' folder."""
    data = {
        "experiments": []
    }

    # Iterate over experiments
    for experiment_id in os.listdir(mlruns_path):
        experiment_path = os.path.join(mlruns_path, experiment_id)

        if not os.path.isdir(experiment_path) or experiment_id == "models":
            continue

        experiment_meta_file = os.path.join(experiment_path, "meta.yaml")       # Load experiment metadata
        experiment_meta = load_yaml(experiment_meta_file) if os.path.exists(experiment_meta_file) else {}

        experiment_data = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_meta.get("name", ""),
            "runs": []
        }

        for run_id in os.listdir(experiment_path):              # Iterate over runs within the experiment
            run_path = os.path.join(experiment_path, run_id)

            if not os.path.isdir(run_path):
                continue

            run_meta_file = os.path.join(run_path, "meta.yaml")     # Load run metadata
            run_meta = load_yaml(run_meta_file) if os.path.exists(run_meta_file) else {}


            metrics = {}    # Load metrics, params, and tags
            metrics_path = os.path.join(run_path, "metrics")
            if os.path.isdir(metrics_path):
                for metric_file in os.listdir(metrics_path):
                    with open(os.path.join(metrics_path, metric_file), 'r') as file:
                        metrics[metric_file] = file.read().strip()

            params = {}
            params_path = os.path.join(run_path, "params")
            if os.path.isdir(params_path):
                for param_file in os.listdir(params_path):
                    with open(os.path.join(params_path, param_file), 'r') as file:
                        params[param_file] = file.read().strip()

            tags = {}
            tags_path = os.path.join(run_path, "tags")
            if os.path.isdir(tags_path):
                for tag_file in os.listdir(tags_path):
                    with open(os.path.join(tags_path, tag_file), 'r') as file:
                        tags[tag_file] = file.read().strip()


            artifacts = []  # Load artifacts
            artifacts_path = os.path.join(run_path, "artifacts")
            if os.path.isdir(artifacts_path):
                for artifact_file in os.listdir(artifacts_path):
                    artifacts.append(artifact_file)

            run_data = {
                "run_id": run_id,
                "run_name": run_meta.get("run_name", ""),
                "metrics": metrics,
                "params": params,
                "tags": tags,
                "artifacts": artifacts
            }
            experiment_data["runs"].append(run_data)
        data["experiments"].append(experiment_data)

    return data

def save_data_to_json(data, output_file="models/mlruns_data.json"):
    """Save the data to a local JSON file."""
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"# utils.py - Data successfully loaded to {output_file}")

def start_mlflow_ui(host="localhost", port=5001):
    """
    Start the MLflow server in a new process.
    :param host: Host to run the MLflow server on.
    :param port: Port to run the MLflow server on.
    """
    command = [
        "mlflow", "server",
        "--default-artifact-root", "artifacts/",
        "--host", host,
        "--port", str(port)
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"MLflow UI started on http://{host}:{port}\n")
    webbrowser.open(f"http://{host}:{port}")