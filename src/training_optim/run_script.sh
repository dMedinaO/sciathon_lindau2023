#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -J training_concat
#SBATCH --array=1-999
#SBATCH -o prueba_%A_%a.out
#SBATCH -e prueba_%A_%a.err
#SBATCH --mem=8gb
#-----------------Módulos---------------------------
module load miniconda3
source activate p45_method
# ----------------Comandos--------------------------
file=$(ls *.txt | sed -n ${SLURM_ARRAY_TASK_ID}p)
python3 training_full_df.py $file