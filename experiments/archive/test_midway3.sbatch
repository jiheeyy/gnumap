#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --account=pi-cdonnat
#SBATCH --array=1-100
#SBATCH --time=20:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G #10G
#SBATCH --mail-type=ALL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jiheeyou@rcc.uchicago.edu  # mail notification for the job


# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "My SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Add lines here to run your computations
job_id=$SLURM_ARRAY_JOB_ID
module load python
source activate gnumap

result_file="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "result file is ${result_file}"
cd $SCRATCH/$USER/gnumap/experiments

if [ ! -d "results/$2" ]; then
    mkdir -p "results/$2"
fi

if [ "$3" = "second" ]; then
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm DGI VGAE GAE GRACE PCA LaplacianEigenmap Isomap TSNE UMAP DenseMAP
    else
        python experiment_main.py --name_dataset "$1" --filename "$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm DGI VGAE GAE GRACE PCA LaplacianEigenmap Isomap TSNE UMAP DenseMAP
    fi
else
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm GNUMAP2 SPAGCN BGRL CCA-SSG
    else
        python experiment_main.py --name_dataset "$1" --filename "$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm GNUMAP2 SPAGCN BGRL CCA-SSG
    fi
fi