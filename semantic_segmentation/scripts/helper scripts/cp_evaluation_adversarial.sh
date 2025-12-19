#!/usr/bin/env bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il,gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=cp_evaluation_adversarial
#SBATCH --output=slurm/evidential/cp_evaluation_adversarial.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd $MMSEG_CONFIGS/..

module load devel/cuda/11.8

python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/at_configs/segformer_mit-b0_8xb4-160k_cityscapes-1024x1024_test.py ../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth work_dirs/cp_calibration/default_segformer/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 1 --workers 2 --out ./work_dirs/cp_eval/train_1-test_2

python - <<'PY'
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
print("Freed CUDA cache")
PY

python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/at_configs/segformer_mit-b0_8xb4-160k_cityscapes-1024x1024_test.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/segformer_adv_trained/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 1 --workers 2 --out ./work_dirs/cp_eval/train_2-test_2

python - <<'PY'
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
print("Freed CUDA cache")
PY

python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/at_configs/segformer_mit-b0_8xb4-160k_cityscapes-1024x1024_test.py ../checkpoints/segformer/segformer_mit-b0_8xb1_1024x1024_160k_cityscapes_cp_trained.pth work_dirs/cp_calibration/segformer_cp_weighted/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 1 --workers 2 --out ./work_dirs/cp_eval/train_3-test_2

python - <<'PY'
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
print("Freed CUDA cache")
PY

python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/at_configs/segformer_mit-b0_8xb4-160k_cityscapes-1024x1024_test.py ../checkpoints/segformer/segformer_mit-b0_8xb2_1024x0124_160k_cityscapes_cp_cospgd_trained.pth work_dirs/cp_calibration/segformer_cp_weighted_adv_trained/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 1 --workers 2 --out ./work_dirs/cp_eval/train_4-test_2


end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime