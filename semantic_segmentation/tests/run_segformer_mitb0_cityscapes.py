from pathlib import Path
import sys
sys.path.append("/home/ma/ma_ma/ma_dschader/workspaces/pfs7wor9/ma_dschader-master-thesis/benchmarking_reliability_generalization/semantic_segmentation")

from semsegbench.utils import get_args_parser
from mmsegmentation.tools.test import main
from mmseg.apis import init_model, inference_model, show_result_pyplot
from torchmetrics.classification import MulticlassJaccardIndex
import pdb

args = get_args_parser()
config_path = Path("configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py")
checkpoint_path = Path("../checkpoints/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth")
img_path = Path("data/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png")


model = init_model(str(config_path), str(checkpoint_path))
result = inference_model(model, str(img_path))
pdb.set_trace()
# vis_image = show_result_pyplot(model, str(img_path), result, save_dir = "../tests", out_file = "result.png", show = False)

result2 = inference_model(model, str(img_path))

seg = result.pred_sem_seg.data
seg2 = result2.pred_sem_seg.data

compared = (seg == seg2)
print("In test mode equal? ->", compared.all())

# activate dropout during inference
model.train()


result1 = inference_model(model, str(img_path))
vis_image = show_result_pyplot(model, str(img_path), result1, save_dir = "../tests", out_file = "../test/result1.png", show = False)
result2 = inference_model(model, str(img_path))
vis_image = show_result_pyplot(model, str(img_path), result2, save_dir = "../tests", out_file = "../test/result2.png", show = False)
seg1 = result1.pred_sem_seg.data
seg2 = result2.pred_sem_seg.data

compared = (seg1 == seg2)
print("In train mode equal? ->", compared.all())

portion = 1-(compared.count_nonzero()/compared.numel())
print("different pixels portion", portion)

metric = MulticlassJaccardIndex(num_classes=19)
iou = metric(seg1.flatten().cpu(), seg2.flatten().cpu())
print("compare seg1 with seg2- iou:", iou)

print("seg1 vs seg - iou:", metric(seg.flatten().cpu(), seg1.flatten().cpu()))
print("seg2 vs seg - iou:", metric(seg.flatten().cpu(), seg2.flatten().cpu()))

# compare seg1 and seg2 with result

pdb.set_trace()
print("end")