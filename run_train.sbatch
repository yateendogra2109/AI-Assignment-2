#!/bin/bash
#SBATCH --job-name=ev_ppo
#SBATCH --partition=btech
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=ev_ppo_%j.out
#SBATCH --error=ev_ppo_%j.err
# The GPU is requested only because the partition format uses it; PPO here runs on the CPU.

# Usage:  sbatch run_train.sbatch            (seed 0)
#         sbatch run_train.sbatch 1          (seed 1)  -> models/ppo_ev_s1
SEED=${1:-0}
if [ "$SEED" = "0" ]; then OUT=models/ppo_ev; else OUT=models/ppo_ev_s${SEED}; fi

cd /scratch/b23cs1082/AI || exit 1

# create the environment once, reuse afterwards
if [ ! -d .venv ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python train.py --timesteps 1500000 --n-envs 4 --seed "$SEED" --out "$OUT"
echo "finished: $OUT.zip"
