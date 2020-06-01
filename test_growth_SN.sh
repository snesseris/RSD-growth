#!/bin/sh
#SBATCH -J varangian\m/
#SBATCH --partition=batch
#SBATCH --get-user-env
#SBATCH -o ./job.%j.out
#SBATCH -e ./job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 3-00:00:00

export OMP_NUM_THREADS=8

srun python montepython/MontePython.py run --superupdate 20 -p input/growth_SN.param -o chains/growth -N 20000