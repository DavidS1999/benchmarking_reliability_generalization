#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il,gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=adv_train_cityscapes_segformer_mitb0
#SBATCH --output=slurm/evidential/adv_train_cityscapes_segformer_mitb0.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd mmsegmentation

module load devel/cuda/11.8

python tools/train.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py --work-dir work_dirs/adv_cospgd_3itr_segformer_mit-b0_8xb1-160k_cityscapes-1024x1024

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime