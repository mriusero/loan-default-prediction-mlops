# Loan Default Prediction with MLOps in Retail Banking

This project carried out in a learning context develops a predictive model to assess the default risk of personal loans in retail banking. By implementing an MLOps pipeline, we ensure efficient, scalable, and reliable deployment of the best-performing model on Amazon Web Services (AWS) using Streamlit.

[See Overview](project-overview.pdf)

## Objectives

Personal loans represent a substantial revenue source for retail banks but carry significant default risks. The objective of this project is to:
- Build a predictive model that estimates each customer's probability of default.
- Develop a comprehensive MLOps pipeline for model deployment and monitoring on AWS.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Tools & Technologies](#tools--technologies)
- [MLOps Lifecycle Overview](#mlops-lifecycle-overview)
- [Experiment Tracking & Management](#experiment-tracking--management)
- [Model Deployment](#model-deployment)
- [Conclusion](#conclusion)
- [License](#license)

## Project Structure

The project follows a modular structure, with the following directories and files:

```
MLOps
.
├── Dockerfile             # Docker configuration file
├── LICENSE                # License information
├── README.md              # Project documentation
├── app.py                 # Main application entry point
├── data                   # Directory for raw and processed data
├── notebooks              # Jupyter Notebooks for EDA
├── poetry.lock            # Poetry lock file for dependencies
├── pyproject.toml         # Poetry configuration file
├── saved_models           # Directory to save trained models with pickle
├── src                    # Source code 
│   ├── app                # Streamlit app layout & components
│   ├── data               # Data loading and processing 
│   ├── features           # Feature engineering 
│   ├── models             # Model development 
│   └── visualization      # Visualization
└── tests                  # Unit test files
```


## Prerequisites

Before you begin, ensure you have the following software installed on your system:

- **Python 3.12+**: This project uses Python, so you'll need to have Python installed. You can download it from [python.org](https://www.python.org/).
- **Poetry**: This project uses Poetry for dependency management. Install it by following the instructions at [python-poetry.org](https://python-poetry.org/docs/#installation).
- **Docker** (Optional): If you prefer to run the project in a Docker container, ensure Docker is installed. Instructions can be found at [docker.com](https://www.docker.com/).

## Installation

Follow these steps to install and set up the project:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mriusero/projet-sda-mlops
   cd projet-sda-mlops
   ```

2. **Install Dependencies:**

   Using Poetry:

   ```bash
   poetry install
   ```

   This will create a virtual environment and install all dependencies listed in `pyproject.toml`.

3. **Activate the Virtual Environment:**

   If Poetry does not automatically activate the virtual environment, you can activate it manually:

   ```bash
   poetry shell
   ```

## Usage

You can run the application locally or inside a Docker container.

### Running Locally

To run the application locally, execute the following command:

```bash
 streamlit run app.py
```

### Running with Docker

1. **Build the Docker Image:**

   ```bash
   docker build -t streamlit .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 8501:8501 streamlit
   ```
   
This will start the application, and you can access it in your web browser at `http://localhost:8501`.

## Tools & Technologies

- **MLflow**: For tracking, managing, and visualizing model performance over time.
- **Streamlit**: Web app framework for deploying the model in an interactive dashboard.
- **AWS**: Cloud platform for scalability and secure hosting (specifically AWS ECR).
- **Git & Docker**: For version control and consistent, containerized deployment.

## MLOps Lifecycle Overview

This project follows an end-to-end MLOps lifecycle, organized as follows:

1. **Business Context**: High default rates on personal loans threaten the bank's revenue.
2. **Metrics for Success**: Key metrics include AUC-ROC, Precision-Recall AUC, F1-Score, Recall, and Precision.
3. **MLOps Stages**:
   - **Data Collection & Preprocessing**: Initial data preparation and cleansing.
   - **Exploratory Data Analysis (EDA)**: Outlier detection, variable relationships, and data quality analysis.
   - **Feature Engineering**: Creating features such as the Debt-to-Income Ratio and handling missing values.
   - **Model Selection**: Testing Random Forest, XGBoost, and LightGBM classifiers.
   - **Hyperparameter Tuning**: Using Optuna for model optimization.

## Experiment Tracking & Management

- **MLflow**: Tracks each experiment, capturing metrics such as Accuracy, F1 Score, Precision, and Recall. This enables easier version control and experiment reproducibility.
- **Optuna**: Used for hyperparameter tuning to improve model performance.

## Model Deployment

- **Containerization**: The model is containerized using Docker for consistent deployment across environments.
- **AWS ECR**: The containerized model is deployed on AWS Elastic Container Registry for secure storage and scalability.
- **CI/CD Pipeline**: Automated with GitHub Actions, including steps for code building, testing, containerization, and deployment to AWS.

## Conclusion

- **Improved Efficiency**: Automation of training, deployment, and monitoring.
- **Enhanced Collaboration**: Facilitates teamwork between data science and DevOps.
- **Accelerated Deployment**: Model deployment time reduced from weeks to hours.
- **Error Reduction**: Consistent environments with Docker and CI/CD.

This project demonstrates the effectiveness of MLOps for developing and deploying robust, scalable, and explainable machine learning solutions in retail banking.

## License

This project is licensed under the terms of the [MIT License](LICENSE).