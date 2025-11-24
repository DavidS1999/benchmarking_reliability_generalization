import pdb


import os, os.path as osp, json, argparse, torch
from torchvision.utils import save_image

from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.registry import DATASETS
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

from cp_utils import preprocess_batch, build_sets_from_probs

def build_dataset(ds_cfg):
    ds = DATASETS.build(ds_cfg)
    if hasattr(ds, "full_init"):
        ds.full_init()
    return ds

def build_loader(ds, batch_size=2, num_workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=pseudo_collate, pin_memory=True, drop_last=False)

def compute_miou_per_image(pred, gt, num_classes, ignore_index=255):
    """calculate mIoU for a single image."""
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)

    # Ignore-Label
    valid = gt != ignore_index
    if not np.any(valid):
        return float("nan")

    pred = pred[valid]
    gt = gt[valid]

    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        ious.append(inter / (union + 1e-10))

    if not ious:
        return float("nan")
    return float(np.mean(ious))

def save_seg_overlay(img_path, seg_pred, palette, alpha, out_folder, miou):
    """saves overlay: original image + color-coded segmentation with alpha blending."""
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]

    # seg_pred: [H,W] (int Labels)
    seg_pred = seg_pred.astype(np.int32)

    # if shapes don't match -> resize
    if seg_pred.shape != (h, w):
        seg_pred_img = Image.fromarray(seg_pred.astype(np.uint8), mode="L")
        seg_pred_img = seg_pred_img.resize((w, h), resample=Image.NEAREST)
        seg_pred = np.array(seg_pred_img)

    # palette: shape [num_classes, 3]
    palette = np.array(palette, dtype=np.uint8)
    color_mask = palette[seg_pred]   # [H,W,3]

    # Alphablending
    overlay = (1 - alpha) * img.astype(np.float32) + alpha * color_mask.astype(np.float32)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    stem = Path(img_path).stem
    out_path = osp.join(out_folder, stem + "_seg_overlay.png")

    plt.figure(figsize=(10,5))
    plt.imshow(overlay)
    title = "Input Image with segmentation mask"
    title += "\n" + r"$\mathrm{mIoU} = " + f"{miou:.4f}$"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"saved segmentation overlay to {out_path}")

def save_heatmap(heatmap, img_meta, alpha, q_hat, out_folder):
    plt.figure(figsize=(10,5))
    stem = Path(img_meta["img_path"]).stem
    im = plt.imshow(heatmap, cmap = "jet", vmin = 0, vmax = 1)
    title = rf"{stem}" + "\n "
    title += r"$\alpha_{\mathrm{cal}}=" +rf"{alpha}$"+", "
    title += r"$\hat{q}" + rf" = {round(q_hat, 8)}$"
    plt.title(title)
    plt.colorbar(im, label = "CP-uncertainty",fraction=0.064)
    plt.tight_layout()
    fname = osp.join(out_folder, stem + f"_qhat={q_hat}.png")
    plt.savefig(fname)
    print(f"saved heatmap to {fname}")
    plt.close()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="inference with Conformal Prediction (global q_hat)")
    ap.add_argument("config", type=str)
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("qhat_json", type=str, help="path to saved q_hat JSON")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=str, default="./work_dirs/cp_infer")
    ap.add_argument("--save-heatmaps", action="store_true", help="save heatmap png")
    ap.add_argument("--save-overlays", action="store_true",
                    help="save RGB image with segmentation overlay")
    ap.add_argument("--overlay-alpha", type=float, default=0.5,
                    help="alpha for segmentation overlay (0..1)")
    args = ap.parse_args()

    with open(args.qhat_json) as f:
        meta = json.load(f)
    q_hat = float(meta["q_hat"])
    alpha = float(meta["alpha"])


    cfg = Config.fromfile(args.config)

    cfg.model.setdefault('test_cfg', {})
    cfg.model.test_cfg['output_logits'] = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg, args.checkpoint, device=device)

    dl_key = f"{args.split}_dataloader"
    ds_cfg = getattr(cfg, dl_key)["dataset"]
    dataset = build_dataset(ds_cfg)
    loader = build_loader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    if hasattr(dataset, "metainfo"):
        metainfo = dataset.metainfo
    elif hasattr(dataset, "METAINFO"):
        metainfo = dataset.METAINFO
    else:
        raise RuntimeError("Dataset has no metainfo/METAINFO with palette/classes")

    palette = metainfo["palette"]
    num_classes = len(palette)

    ignore_index = getattr(getattr(model, "decode_head", None), "ignore_index", 255)

    os.makedirs(args.out, exist_ok=True)
    total_pix = 0
    set1_pix = 0

    for i, batch in enumerate(loader):
        
        inputs = batch["inputs"]
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs, dim=0)  # [B,3,H,W]
        inputs = inputs.to(device).contiguous()

        data_samples = batch["data_samples"]

        
        for ds in data_samples:
            if hasattr(ds, "gt_sem_seg") and ds.gt_sem_seg is not None:
                ds.gt_sem_seg.data = ds.gt_sem_seg.data.to(device, non_blocking=True)

        pred_samples = model.predict(inputs, data_samples)

        logit_list = []
        for ds in pred_samples:
            if hasattr(ds, "seg_logits") and ds.seg_logits is not None:
                logit_list.append(ds.seg_logits.data)  # [C,H,W]
            else:
                raise RuntimeError(
                    "predict() has no seg_logits. "
                    "set in config: cfg.model.test_cfg['output_logits'] = True"
                )

        logits = torch.stack(logit_list, dim=0)   # [B,C,H,W]
        probs  = torch.softmax(logits, dim=1)
        imgs, _, img_metas = preprocess_batch(model, batch)

        # this part works without attacks, because encode_decode isn't using the predict method with the attack code
        # logits = model.encode_decode(imgs, img_metas)

        set_mask, set_size = build_sets_from_probs(probs, q_hat)
        pred_labels = probs.argmax(dim=1)

        total_pix += set_size.numel()
        set1_pix += (set_size == 1).sum().item()

        if args.save_overlays:
            data_samples = batch["data_samples"]

            for b in range(pred_labels.size(0)):
                seg_pred_np = pred_labels[b].cpu().numpy().astype(np.uint8)

                gt_sem_seg = data_samples[b].gt_sem_seg.data
                if isinstance(gt_sem_seg, torch.Tensor):
                    gt_np = gt_sem_seg.squeeze().cpu().numpy().astype(np.int64)
                else:
                    gt_np = gt_sem_seg.numpy().squeeze().astype(np.int64)
                
                if seg_pred_np.shape != gt_np.shape:
                    seg_pred_img = Image.fromarray(seg_pred_np, mode="L")
                    h_gt, w_gt = gt_np.shape
                    seg_pred_img = seg_pred_img.resize((w_gt, h_gt), resample=Image.NEAREST)
                    seg_pred_np = np.array(seg_pred_img)
                
                miou = compute_miou_per_image(seg_pred_np, gt_np,
                                              num_classes=num_classes,
                                              ignore_index=ignore_index)


                img_path = img_metas[b]["img_path"]
                save_seg_overlay(img_path, seg_pred_np, palette,
                                 alpha=args.overlay_alpha, out_folder=args.out, miou=miou)

        if args.save_heatmaps:
            ss = set_size.float()
            num_classes = probs.size(1)
            ss = (ss - 1) / (num_classes - 1) # uncertainty normalized to [0,1] -> 0 equals set size 1, 1 equals set size = num_classes
            ss = ss.squeeze(dim=1)
            for b in range(ss.size(0)):
                
                save_heatmap(ss[b].cpu().numpy(), img_metas[b], alpha, q_hat, args.out)
                

            # old code for saving heatmaps without normalization
            # for b in range(set_size.shape[0]):
            #     import pdb
            #     pdb.set_trace()
            #     save_heatmap(set_size[b,0].cpu().numpy(), img_metas[b][b], q_hat, args.out)
        
        if i == 1:
            break
                
           

    print(f"Share of pixel with set size==1: {set1_pix/total_pix:.4f} ({set1_pix}/{total_pix})")
    print(f"q_hat (alpha≈{1-float(meta['alpha']):.2f} coverage target): {q_hat:.6f}")

if __name__ == "__main__":
    main()
