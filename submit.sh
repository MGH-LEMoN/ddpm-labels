#!/bin/bash
#SBATCH --account=lcnrtx
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output="./logs/slurm-logs/%x.out"
#SBATCH --error="./logs/slurm-logs/%x.err"
#SBATCH --mail-user=hvgazula@umich.edu
#SBATCH --mail-type=FAIL

source /space/calico/1/users/Harsha/anaconda3/etc/profile.d/conda.sh
conda activate ddpm

echo 'Start time:' `date`
echo 'Node:' $HOSTNAME
echo "$@"
start=$(date +%s)

"$@"

end=$(date +%s)
echo 'End time:' `date`
echo "Elapsed Time: $(($end-$start)) seconds"
