cd $MMSEG_CONFIGS/..

python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b0_8xb4-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth work_dirs/cp_calibration/default_segformer/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/train_1-test_1


python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024-at.py ../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth work_dirs/cp_calibration/default_segformer/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/train_1-test_3




python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b0_8xb4-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/segformer_adv_trained/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/train_2-test_1



python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024-at.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/segformer_adv_trained/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/train_2-test_3




python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b0_8xb4-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b0_8xb1_1024x1024_160k_cityscapes_cp_trained.pth work_dirs/cp_calibration/segformer_cp_weighted/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/train_3-test_1


python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024-at.py ../checkpoints/segformer/segformer_mit-b0_8xb1_1024x1024_160k_cityscapes_cp_trained.pth work_dirs/cp_calibration/segformer_cp_weighted/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/train_3-test_3



python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b0_8xb4-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b0_8xb2_1024x0124_160k_cityscapes_cp_cospgd_trained.pth work_dirs/cp_calibration/segformer_cp_weighted_adv_trained/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/train_4-test_1

python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024-at.py ../checkpoints/segformer/segformer_mit-b0_8xb2_1024x0124_160k_cityscapes_cp_cospgd_trained.pth work_dirs/cp_calibration/segformer_cp_weighted_adv_trained/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/train_4-test_3



python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth work_dirs/cp_calibration/default_segformer/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/mit-b1


python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth work_dirs/cp_calibration/default_segformer/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/mit-b2


python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth work_dirs/cp_calibration/default_segformer/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/mit-b3

python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth work_dirs/cp_calibration/default_segformer/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/mit-b4


python tools/cp/evaluate_cp_uncertainty.py ../configs/segformer/default/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth work_dirs/cp_calibration/default_segformer/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --out ./work_dirs/cp_eval/mit-b5