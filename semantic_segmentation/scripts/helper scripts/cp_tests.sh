cd $MMSEG_CONFIGS/..

python tools/cp/infer_cp.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.1.json --split val --batch-size 2 --workers 2 --save-heatmaps --save-overlays --overlay-alpha 0.4 --out work_dirs/cp_infer/trained-default_tested-default

python tools/cp/infer_cp.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 2 --workers 2 --save-heatmaps --out work_dirs/cp_infer/trained-default_tested-default

python tools/cp/infer_cp.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py ../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.4.json --split val --batch-size 2 --workers 2 --save-heatmaps --out work_dirs/cp_infer/trained-default_tested-default