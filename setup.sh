#!/bin/bash

# Set the project name and Python version
PROJECT_NAME="phy_399"  # Replace with your project name
PYTHON_VERSION="3.11"

# CUDA and cuDNN versions
CUDA_VERSION="11.2"
CUDNN_VERSION="8.1.0"

#NO LONGER NEEDED SINCE I REMOVED PIPENV AND NONE OF YOU HAVE IT 
# # Function to deactivate any active virtual environments
# deactivate_virtualenv() {
#     if [[ -n "$VIRTUAL_ENV" ]]; then
#         echo "Deactivating the current virtual environment: $VIRTUAL_ENV"
#         deactivate
#     elif [[ -n "$CONDA_PREFIX" ]]; then
#         echo "Deactivating the current Conda environment: $CONDA_PREFIX"
#         conda deactivate
#     else
#         echo "No virtual environment is currently active"
#     fi
# }

# # Deactivate any active virtual environments
# deactivate_virtualenv

# # Export Pipenv dependencies to requirements.txt
# echo "Exporting Pipenv dependencies to requirements.txt..."
# pipenv lock -r > requirements.txt

# # Remove the old Pipenv environment
# echo "Removing the old Pipenv environment..."
# pipenv --rm

# # Delete Pipenv-related files
# echo "Deleting Pipenv-related files..."
# rm -f Pipfile Pipfile.lock

# INSTAL CUDA for wsl2 and nvida gpu's with pytorch support
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# INSTALL CONDA ON ubuntu
echo "installing miniconda"
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

echo "initalizing miniconda" 
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh


# Create a new Conda environment with CUDA support from environment.yml
echo "Creating conda environment from environment.yml"
conda env create -f environment.yml
# echo "Creating a new Conda environment named $PROJECT_NAME with Python $PYTHON_VERSION and CUDA $CUDA_VERSION..."
# conda create --name $PROJECT_NAME python=$PYTHON_VERSION -y
echo "Activating ENV"
conda activate Ultrasound_Physics_Simulator


# Add the necessary channels
conda config --add channels conda-forge
conda config --add channels defaults
conda config --add channels nvidia

# Install CUDA and cuDNN
echo "Installing CUDA $CUDA_VERSION and cuDNN $CUDNN_VERSION..."
conda install -c conda-forge cudatoolkit=$CUDA_VERSION cudnn=$CUDNN_VERSION -y

# # Install TensorFlow with GPU support - IN GODS NAME DO NOT USE THIS IT DOESNT WORK ANYMORE
# echo "Installing TensorFlow with GPU support..."
# conda install -c conda-forge tensorflow-gpu -y

# Install PyTorch with GPU support - THIS IS THEONE THATS IMPORTANT
echo "Installing PyTorch with GPU support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

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

echo "Getting Latest Version Of Conda" 
conda update conda -y

echo "Migration complete and old Pipenv and virtual environments removed!"