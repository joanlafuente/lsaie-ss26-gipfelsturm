#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:10:00
#SBATCH --partition=debug
#SBATCH --job-name=setup-env
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue

echo "START TIME: $(date)"
sleep 10  # Simulate some setup time

# Use srun to launch the Pyxis container, then pass a multi-line string 
# to 'bash -c' so that all these commands run INSIDE the container.
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=/users/$USER/lsaie-ss26-gipfelsturm/alps3.toml --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "
cd /users/$USER/lsaie-ss26-gipfelsturm
echo 'Python version inside container:'
python --version

# 1. Create the environment
python -m venv --system-site-packages my_env

# 2. Activate it
source my_env/bin/activate

# 3. Upgrade pip (crucial for pyproject.toml files)
pip install --upgrade pip

# 4. Install your packages (quoted so bash doesn't break on the brackets)
pip install peft wandb 'cut-cross-entropy[transformers] @ git+https://github.com/apple/ml-cross-entropy.git'
"

echo "END TIME: $(date)"