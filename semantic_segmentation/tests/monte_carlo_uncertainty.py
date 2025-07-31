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

# from VALUES
def calculate_uncertainty(softmax_preds: torch.Tensor, ssn: bool = False):
    uncertainty_dict = {}
    # softmax_preds = torch.from_numpy(softmax_preds)
    mean_softmax = torch.mean(softmax_preds, dim=0)
    pred_entropy = torch.zeros(*softmax_preds.shape[2:], device=mean_softmax.device)
    for y in range(mean_softmax.shape[0]):
        pred_entropy_class = mean_softmax[y] * torch.log(mean_softmax[y])
        nan_pos = torch.isnan(pred_entropy_class)
        pred_entropy[~nan_pos] += pred_entropy_class[~nan_pos]
    pred_entropy *= -1
    expected_entropy = torch.zeros(
        softmax_preds.shape[0], *softmax_preds.shape[2:], device=softmax_preds.device
    )
    for pred in range(softmax_preds.shape[0]):
        entropy = torch.zeros(*softmax_preds.shape[2:], device=softmax_preds.device)
        for y in range(softmax_preds.shape[1]):
            entropy_class = softmax_preds[pred, y] * torch.log(softmax_preds[pred, y])
            nan_pos = torch.isnan(entropy_class)
            entropy[~nan_pos] += entropy_class[~nan_pos]
        entropy *= -1
        expected_entropy[pred] = entropy
    expected_entropy = torch.mean(expected_entropy, dim=0)
    mutual_information = pred_entropy - expected_entropy

    # pdb.set_trace()
    n_classes = softmax_preds.shape[1]
    normalized_mi = mutual_information / torch.log(torch.tensor(n_classes, device = mutual_information.device))

    uncertainty_dict["pred_entropy"] = pred_entropy
    if not ssn:
        uncertainty_dict["aleatoric_uncertainty"] = expected_entropy
        uncertainty_dict["epistemic_uncertainty"] = mutual_information
    else:
        print("mutual information is aleatoric unc")
        uncertainty_dict["aleatoric_uncertainty"] = mutual_information
        uncertainty_dict["epistemic_uncertainty"] = expected_entropy
    # value["softmax_pred"] = np.mean(value["softmax_pred"], axis=0)
    return uncertainty_dict

def activate_dropout(model):
    for layer in model.modules():
        if layer.__class__.__name__.startswith("Dropout"):
            layer.train()

def plot_imshow(array, fig_name):
    plt.figure()
    plt.imshow(array, cmap = "gray")
    plt.colorbar()
    plt.savefig(f"../tests/imgs/{fig_name}.png")
    plt.close()

def get_uncertainty_map(model, img_path, n_runs:int = 11, return_logits = False):
    
    softmax_preds = []
    pred_sem_segs = []
    logits = []
    
    for _ in tqdm(range(n_runs), desc = "predicting..."):
        result = inference_model(model, str(img_path))
        probs = torch.softmax(result.seg_logits.data, dim = 0)
        softmax_preds.append(probs)
        pred_sem_segs.append(result.pred_sem_seg.data)
        logits.append(result.seg_logits.data)
    
    softmax_preds = torch.stack(softmax_preds)
    uncertainty_dict = calculate_uncertainty(softmax_preds)

    uc_map = uncertainty_dict["epistemic_uncertainty"]
    if return_logits:
        return uc_map, logits
    else:
        return uc_map

args = get_args_parser()
config_path = Path("configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py")
checkpoint_path = Path("../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth")
img_path = Path("data/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png")


model = init_model(str(config_path), str(checkpoint_path))
model.eval()
activate_dropout(model)

uc_map, logits = get_uncertainty_map(model, img_path, return_logits = True)
pdb.set_trace()

# e = F.softplus(logits)
# alpha = e + 1
# S = torch.sum(alpha, dim=0, keepdim=True)
# probs = alpha/S
# uncertainty = 19 / (S+1)

uc_differences = {"uc_map": [], "uc_max": [], "uc_argmax": [], "uc_mean": []}
for i in range(10):
    uc_map = get_uncertainty_map(model, img_path)

    uc_max = uc_map.max()
    uc_argmax = (uc_map == uc_max).nonzero()
    uc_mean = uc_map.mean()

    uc_differences["uc_map"].append(uc_map)
    uc_differences["uc_max"].append(uc_max.item())
    pdb.set_trace()
    uc_differences["uc_argmax"].append(uc_argmax.cpu().numpy())
    uc_differences["uc_mean"].append(uc_mean.item())

print("ux_max:", uc_differences["uc_max"])
print("uc_argmax:", uc_differences["uc_argmax"])
print("uc_mean:", uc_differences["uc_mean"])

pdb.set_trace()
print("end")