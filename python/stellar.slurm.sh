#!/bin/bash
#SBATCH --job-name=hst       # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=24              # total number of tasks
##SBATCH --exclusive
#SBATCH --cpus-per-task=1        # cpu-cores per task
##SBATCH --mem=3000G
##SBATCH --time=01:00:00
##SBATCH --partition=bigmem
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2022.5 openmpi/gcc/4.1.0
conda activate pyathena-lem

export PYTHONPATH="$HOME/pyathena"
echo $PYTHONPATH

cd $HOME/pyathena/working_notebook/TIGRESS-NCR/lowZ/TIGRESS-NCR-fitting/
# srun python coolheat_load_save.py
srun python gravity_stress.py 1> out.txt 2> err.txt
#srun python pyathena/tigress_ncr/do_tasks_chbreak.py -b $OUTDIR 1> scripts/do_tasks-$SLURM_JOB_ID.out 2> scripts/do_tasks-$SLURM_JOB_ID.err
