#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --output=setup_%j.out

# Load modules
module purge
module load anaconda3/2020.07

# Remove old environment if exists
rm -rf /scratch/am14419/conda_envs/hw4-nlp

# Create conda environment in scratch
echo "Creating conda environment..."
conda create --prefix /scratch/am14419/conda_envs/hw4-nlp python=3.9 pip -y

# Activate environment
source activate /scratch/am14419/conda_envs/hw4-nlp

# Verify paths
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"

# Upgrade pip
python -m pip install --upgrade pip

# Install packages
echo "Installing packages..."
python -m pip install --no-cache-dir \
    transformers==4.26.1 \
    datasets==2.9.0 \
    evaluate==0.4.0 \
    scikit-learn==1.2.1 \
    nltk==3.8.1 \
    torch==1.13.1

# Create NLTK data directory
mkdir -p /scratch/am14419/nltk_data

# Download NLTK data
echo "Downloading NLTK data..."
python3 << 'EOF'
import nltk
nltk.download('wordnet', download_dir='/scratch/am14419/nltk_data')
nltk.download('punkt', download_dir='/scratch/am14419/nltk_data')
EOF

# Test installation
echo "Testing installation..."
python3 << 'EOF'
import torch
import transformers
import datasets
import evaluate
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"Package location: {torch.__file__}")
print("\n✓ All packages installed successfully!")
EOF

echo "Setup complete!"
