cd $MMSEG_CONFIGS/..

python tools/cp/infer_cp.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024_test_mode.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.1.json --split val --batch-size 1 --workers 2 --save-heatmaps --save-overlays --overlay-alpha 0.4 --out work_dirs/cp_infer/acdc/trained-cospgd_tested-default

python tools/cp/infer_cp.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024_test_mode.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 1 --workers 2 --save-heatmaps --overlay-alpha 0.4 --out work_dirs/cp_infer/acdc/trained-cospgd_tested-default

python tools/cp/infer_cp.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024_test_mode.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.4.json --split val --batch-size 1 --workers 2 --save-heatmaps --overlay-alpha 0.4 --out work_dirs/cp_infer/acdc/trained-cospgd_tested-default


python tools/cp/infer_cp.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024-at.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.1.json --split val --batch-size 1 --workers 2 --save-heatmaps --save-overlays --overlay-alpha 0.4 --out work_dirs/cp_infer/acdc/trained-cospgd_tested-cospgd

python tools/cp/infer_cp.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024-at.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.2.json --split val --batch-size 1 --workers 2 --save-heatmaps --overlay-alpha 0.4 --out work_dirs/cp_infer/acdc/trained-cospgd_tested-cospgd

python tools/cp/infer_cp.py ../configs/segformer/acdc/segformer_mit-b0_8xb1-160k_acdc-1024x1024-at.py ../checkpoints/segformer/segformer_mit-b0_8xb4_1024x1024_160k_cityscapes_cospgd_trained.pth work_dirs/cp_calibration/cp_qhat_city-frankfurt_val_alpha-0.4.json --split val --batch-size 1 --workers 2 --save-heatmaps --overlay-alpha 0.4 --out work_dirs/cp_infer/acdc/trained-cospgd_tested-cospgd



