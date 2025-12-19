# Code base for Master Thesis "Analyzing Robustness in Uncertainty-Aware Semantic Segmentation under Adversarial Attacks"

Based on SemSegBench by https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/tree/main/semantic_segmentation. 

## Installation

```bash
git clone https://github.com/DavidS1999/benchmarking_reliability_generalization.git
cd benchmarking_reliability_generalization

conda env create -f environment.yml
conda activate semseg

pip install -v -e semantic_segmentation/mmsegmentation

```

# Train Scenarios
For the three train scenarios 
1. Adversarially tarined
2. CP-weighted loss
3. Adversarially trained + CP-weighted loss

the checkpoint files can be found under `semantic_segmentation/checkpoints/segformer`.
For the default traing scenario checkpoint, please download the one from MMSegmentation.

# Conformal Prediction
The main code for Conformal Prediction calibration, inference and evaluation can be found under `semantic_segmentation/mmsegmentation/tools/cp`.

The config files for training the model with CP-weighted loss (and adversarial attacks) can be found under `semantic_segmentation/configs/segformer/cp_configs`.