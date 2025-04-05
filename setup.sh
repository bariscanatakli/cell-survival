#!/bin/bash

echo "Setting up cell-survival-RL environment"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Check if venv is installed
PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1-2)
if python3 -c "import venv" &> /dev/null; then
    echo "venv module is available."
else
    echo "venv module is not available. Installing venv..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y python${PYTHON_VERSION}-venv
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        pip3 install virtualenv
    else
        echo "Unsupported OS. Please install venv manually."
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Ask about GPU support
echo ""
echo "Do you want to install PyTorch with GPU support? (y/n)"
echo "Note: This requires having a compatible NVIDIA GPU with CUDA installed."
read -r use_gpu

if [[ $use_gpu == "y" || $use_gpu == "Y" ]]; then
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1)
        MINOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f2)
        
        echo "Detected CUDA version: $CUDA_VERSION"
        
        if [[ $MAJOR_VERSION -eq 11 && $MINOR_VERSION -eq 7 ]]; then
            echo "Installing PyTorch with CUDA 11.7 support..."
            pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
        elif [[ $MAJOR_VERSION -eq 11 && $MINOR_VERSION -eq 8 ]]; then
            echo "Installing PyTorch with CUDA 11.8 support..."
            pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
        else
            echo "Installing PyTorch with latest CUDA support..."
            pip install torch
        fi
    else
        echo "CUDA not detected. Installing PyTorch with latest available CUDA support..."
        pip install torch
    fi
else
    echo "Installing PyTorch without GPU support..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Run the simulation with:"
echo "source venv/bin/activate"
echo "python run_large_simulation.py"
