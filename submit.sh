#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=lcnrtx,rtx6000,rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output="./logs/slurm-logs/%x.out"
#SBATCH --error="./logs/slurm-logs/%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

conda activate ddpm

echo 'Start time:' `date`
echo 'Node:' $HOSTNAME
echo "$@"
start=$(date +%s)

"$@"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
