from pathlib import Path
import sys
sys.path.append("/home/ma/ma_ma/ma_dschader/workspaces/pfs7wor9/ma_dschader-master-thesis/benchmarking_reliability_generalization/semantic_segmentation")

from semsegbench.utils import get_args_parser
from mmsegmentation.tools.test import main
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg.apis.utils import _preprare_data
from torchmetrics.classification import MulticlassJaccardIndex
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

def plot_imshow(array, fig_name):
    plt.figure()
    plt.imshow(array, cmap = "gray")
    plt.colorbar()
    plt.savefig(f"../tests/imgs/{fig_name}.png")
    plt.close()


args = get_args_parser()
config_path = Path("configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py")
checkpoint_path = Path("../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth")
img_path = Path("data/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png")


model = init_model(str(config_path), str(checkpoint_path))
model.eval()

result = inference_model(model, str(img_path))
seg_logits = result.seg_logits.data
pred_sem_seg = result.pred_sem_seg.data

energy = -torch.logsumexp(seg_logits, dim=0)
energy_norm = (energy-energy.min()) / (energy.max()-energy.min())

# also entropy
probs = torch.softmax(seg_logits, dim = 0)
e = -torch.sum(probs * torch.log(probs+1e-8), dim = 0)
e_norm = (e-e.min()) / (e.max() - e.min())

pdb.set_trace()
print("end")