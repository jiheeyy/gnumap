#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --account=caslake
#SBATCH --array=1-100
#SBATCH --time=20:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=40G #10G
#SBATCH --mail-type=ALL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jiheeyou@rcc.uchicago.edu  # mail notification for the job


# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "My SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Add lines here to run your computations
job_id=$SLURM_ARRAY_JOB_ID
module load python
source activate symbol_gnumap

result_file="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "result file is ${result_file}"
cd $SCRATCH/$USER/gnumap/experiments

if [ ! -d "results/$3_$2" ]; then
    mkdir -p "results/$3_$2"
fi

if [ "$3" = "quickten" ]; then
    # Run the experiment with 'quickten' models
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm DGI VGAE GAE GNUMAP2 PCA LaplacianEigenmap Isomap TSNE UMAP DenseMAP
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm DGI VGAE GAE GNUMAP2 PCA LaplacianEigenmap Isomap TSNE UMAP DenseMAP
    fi
elif [ "$3" = "spagcn" ]; then
    # Run the experiment with 'spagcn' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm SPAGCN
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm SPAGCN
    fi
elif [ "$3" = "bgrl" ]; then
    # Run the experiment with 'bgrl' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm BGRL
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm BGRL
    fi
elif [ "$3" = "grace" ]; then
    # Run the experiment with 'grace' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm GRACE
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm GRACE
    fi
elif [ "$3" = "ccassg" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm CCA-SSG
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm CCA-SSG
    fi
elif [ "$3" = "dgi" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm DGI
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm DGI
    fi
elif [ "$3" = "vgae" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm VGAE
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm VGAE
    fi
elif [ "$3" = "gae" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm GAE
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm GAE
    fi
elif [ "$3" = "gnumap2" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm GNUMAP2
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm GNUMAP2
    fi
elif [ "$3" = "pca" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm PCA
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm PCA
    fi
elif [ "$3" = "le" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm LaplacianEigenmap
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm LaplacianEigenmap
    fi
elif [ "$3" = "isomap" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm Isomap
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm Isomap
    fi
elif [ "$3" = "tsne" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm TSNE
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm TSNE
    fi
elif [ "$3" = "umap" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm UMAP
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm UMAP
    fi
elif [ "$3" = "densemap" ]; then
    # Run the experiment with 'cca-ssg' model
    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 1 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm DenseMAP
    else
        python experiment_main.py --name_dataset "$1" --filename "$3_$2" --save_img 0 --result_file "$result_file" --seed $SLURM_ARRAY_TASK_ID --jm DenseMAP
    fi
else
    echo "ERROR! No model to run"
    exit 1
fi
