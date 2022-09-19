#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --time=35:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --account=pi-cdonnat

# Print the task id.
# Add lines here to run your computations

module load python
source activate pytorch_geom3.9
echo $1
echo $2
job_id=$(sbatch --parsable test.sh)
echo "the job is $job_id"
METH=$1
cd $SCRATCH/$USER/gnumap/experiments
