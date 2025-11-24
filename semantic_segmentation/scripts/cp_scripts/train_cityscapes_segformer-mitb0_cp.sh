#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il,gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=48
#SBATCH --cpus-per-task=8
#SBATCH --job-name=train_cityscapes_segformer_mitb0_cp
#SBATCH --output=slurm/evidential/train_cityscapes_segformer_mitb0_cp.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd mmsegmentation

module load devel/cuda/11.8

python tools/train.py ../configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024_cp.py --work-dir work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024_cp

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime