#!/usr/bin/env bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=train_cityscapes_segformer_mitb0_evidential
#SBATCH --output=slurm/evidential/train_cityscapes_segformer_mitb0_evidential.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd mmsegmentation

module load devel/cuda/11.8

python tools/train.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime