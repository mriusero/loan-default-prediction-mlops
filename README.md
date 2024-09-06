# projet-sda-mlops

This repository contains a streamlit application template designed for MLOps. 

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

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
python streamlit run app.py
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

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any feature requests, bug fixes, or improvements.

1. Fork the Repository
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
```
This project is licensed under the terms of the [MIT License](LICENSE).
```
