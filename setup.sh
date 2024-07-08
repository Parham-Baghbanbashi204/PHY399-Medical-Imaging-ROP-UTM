#!/bin/bash

# Set the project name and Python version
PROJECT_NAME="phy_399"  # Replace with your project name
PYTHON_VERSION="3.11"

# CUDA and cuDNN versions
CUDA_VERSION="11.2"
CUDNN_VERSION="8.1.0"

# Function to deactivate any active virtual environments
deactivate_virtualenv() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "Deactivating the current virtual environment: $VIRTUAL_ENV"
        deactivate
    elif [[ -n "$CONDA_PREFIX" ]]; then
        echo "Deactivating the current Conda environment: $CONDA_PREFIX"
        conda deactivate
    else
        echo "No virtual environment is currently active"
    fi
}

# Deactivate any active virtual environments
deactivate_virtualenv

# Export Pipenv dependencies to requirements.txt
echo "Exporting Pipenv dependencies to requirements.txt..."
pipenv lock -r > requirements.txt

# Remove the old Pipenv environment
echo "Removing the old Pipenv environment..."
pipenv --rm

# Delete Pipenv-related files
echo "Deleting Pipenv-related files..."
rm -f Pipfile Pipfile.lock

# Create a new Conda environment with CUDA support
echo "Creating a new Conda environment named $PROJECT_NAME with Python $PYTHON_VERSION and CUDA $CUDA_VERSION..."
conda create --name $PROJECT_NAME python=$PYTHON_VERSION -y
conda activate $PROJECT_NAME

# Add the necessary channels
conda config --add channels conda-forge
conda config --add channels defaults
conda config --add channels nvidia

# Install CUDA and cuDNN
echo "Installing CUDA $CUDA_VERSION and cuDNN $CUDNN_VERSION..."
conda install -c conda-forge cudatoolkit=$CUDA_VERSION cudnn=$CUDNN_VERSION -y

# Install TensorFlow with GPU support
echo "Installing TensorFlow with GPU support..."
conda install -c conda-forge tensorflow-gpu -y

# Install PyTorch with GPU support
echo "Installing PyTorch with GPU support..."
conda install -c pytorch pytorch torchvision torchaudio cudatoolkit=$CUDA_VERSION -y

# Install other dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify the environment
echo "Verifying the environment..."
conda list

# Export the environment to environment.yml (optional)
echo "Exporting the environment to environment.yml..."
conda env export > environment.yml

# Deactivate the Conda environment
echo "Deactivating the Conda environment..."
conda deactivate

echo "Migration complete and old Pipenv and virtual environments removed!"