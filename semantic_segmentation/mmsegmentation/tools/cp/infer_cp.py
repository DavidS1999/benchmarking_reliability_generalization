# tools/cp/infer_cp.py
import pdb


import os, os.path as osp, json, argparse, torch
from torchvision.utils import save_image

from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.registry import DATASETS
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader


from cp_utils import preprocess_batch, build_sets_from_probs

def build_dataset(ds_cfg):
    ds = DATASETS.build(ds_cfg)
    if hasattr(ds, "full_init"):
        ds.full_init()
    return ds

def build_loader(ds, batch_size=2, num_workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=pseudo_collate, pin_memory=True, drop_last=False)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Inference mit Conformal Prediction (globales q_hat)")
    ap.add_argument("config", type=str)
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("qhat_json", type=str, help="Pfad zur gespeicherten q_hat JSON")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=str, default="./work_dirs/cp_infer")
    ap.add_argument("--save-heatmaps", action="store_true", help="Setgrößen-Heatmaps als PNG speichern")
    args = ap.parse_args()

    with open(args.qhat_json) as f:
        meta = json.load(f)
    q_hat = float(meta["q_hat"])

    cfg = Config.fromfile(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg, args.checkpoint, device=device)

    dl_key = f"{args.split}_dataloader"
    ds_cfg = getattr(cfg, dl_key)["dataset"]
    dataset = build_dataset(ds_cfg)
    loader = build_loader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    os.makedirs(args.out, exist_ok=True)
    total_pix = 0
    set1_pix = 0

    for i, batch in enumerate(loader):
        imgs, _, img_metas = preprocess_batch(model, batch)
        logits = model.encode_decode(imgs, img_metas)
        probs = torch.softmax(logits, dim=1)
        set_mask, set_size = build_sets_from_probs(probs, q_hat)

        # simple metric: Anteil Setgröße==1
        total_pix += set_size.numel()
        set1_pix += (set_size == 1).sum().item()

        if args.save_heatmaps:
            # Normierung: größere Setgröße ⇒ heller
            # [B,1,H,W] → [B,1,H,W] float in [0,1]
            ss = set_size.float()
            ss = (ss - ss.min()) / (ss.max() - ss.min() + 1e-6)
            for b in range(ss.size(0)):
                fname = osp.join(args.out, f"setsize_{i:04d}_{b:02d}.png")
                save_image(ss[b], fname)

    print(f"Anteil Pixel mit Setgröße==1: {set1_pix/total_pix:.4f} ({set1_pix}/{total_pix})")
    print(f"q_hat (alpha≈{1-float(meta['alpha']):.2f} coverage target): {q_hat:.6f}")

if __name__ == "__main__":
    main()
