"""
Evaluate Conformal Prediction (CP) uncertainty maps for semantic segmentation on a dataset.

What it computes (pixel-level, ignoring ignore_index):
- CP validity: coverage P(y in set) and mean set size
- Error-detection quality of CP-uncertainty (set size) for top-1 mistakes:
    - AUROC and AUPRC for detecting error pixels from uncertainty
- Selective prediction / abstention:
    - Risk Coverage curve (keep most certain pixels) + AURC

This is designed to work with MMSegmentation models that expose seg_logits
via cfg.model.test_cfg['output_logits'] = True.
"""

import os, os.path as osp, json, argparse, math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.registry import DATASETS
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader
import pdb

from cp_utils import preprocess_batch, build_sets_from_probs


def build_dataset(ds_cfg):
    ds = DATASETS.build(ds_cfg)
    if hasattr(ds, "full_init"):
        ds.full_init()
    return ds

def build_loader(ds, batch_size=2, num_workers=2):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=pseudo_collate, pin_memory=True, drop_last=False
    )

def safe_div(a, b):
    return float(a) / float(b) if b else float("nan")

def ece_binned(conf: np.ndarray, correct: np.ndarray, n_bins: int = 19):
    """
    ECE-style: bins confidence in [0,1], compares mean confidence vs. empirical accuracy.
    conf:    float array in [0,1], shape [N]
    correct: bool or {0,1} array, shape [N]
    returns: ece (float), plus per-bin diagnostics
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    # clamp just in case of tiny numerical drift
    conf = np.clip(conf, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    bin_acc = np.full(n_bins, np.nan, dtype=np.float64)
    bin_conf = np.full(n_bins, np.nan, dtype=np.float64)
    bin_frac = np.zeros(n_bins, dtype=np.float64)

    N = conf.shape[0]
    if N == 0:
        return float("nan"), (bin_edges, bin_acc, bin_conf, bin_frac)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)

        cnt = m.sum()
        if cnt > 0:
            bin_frac[b] = cnt / N
            bin_conf[b] = conf[m].mean()
            bin_acc[b] = correct[m].mean()
            ece += bin_frac[b] * abs(bin_acc[b] - bin_conf[b])

    return float(ece), (bin_edges, bin_acc, bin_conf, bin_frac)

def auc_trapz(x, y):
    """Trapezoidal area; expects x increasing."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    return float(np.trapz(y, x))

def roc_pr_from_bins(pos_counts_desc, neg_counts_desc):
    """
    Given counts for each threshold bucket sorted by score DESC,
    compute ROC and PR curves.
    Returns:
      fpr, tpr, precision, recall
    """
    pos = np.asarray(pos_counts_desc, dtype=np.float64)
    neg = np.asarray(neg_counts_desc, dtype=np.float64)

    P = pos.sum()
    N = neg.sum()
    # cumulative as we lower threshold (include more pixels)
    tp = np.cumsum(pos)
    fp = np.cumsum(neg)

    tpr = tp / (P + 1e-12)
    fpr = fp / (N + 1e-12)

    precision = tp / (tp + fp + 1e-12)
    recall = tpr  # since recall = TPR

    # ROC padding (optional, but fine)
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])

    # PR padding: prepend (recall=0, precision=1)
    # (this matches sklearn's convention and guarantees equal lengths)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])


    return fpr, tpr, precision, recall

def aurc_from_risk_coverage(coverages, risks):
    """Area under risk-coverage curve (coverage increasing)."""
    return auc_trapz(coverages, risks)

def plot_risk_coverage(coverages, risks, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(coverages, risks)
    plt.xlim([0,1])
    plt.xlabel("Coverage (fraction of pixels kept)")
    plt.ylabel("Risk (error rate among kept pixels)")
    plt.title("Risk–Coverage curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_reliability_discrete(conf_vals, acc_vals, frac_vals, out_path):
    conf_vals = np.asarray(conf_vals, dtype=np.float64)
    acc_vals = np.asarray(acc_vals, dtype=np.float64)
    frac_vals = np.asarray(frac_vals, dtype=np.float64)

    plt.figure(figsize=(5.5, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

    # points (optional: point size proportional to bin fraction)
    sizes = 2000.0 * frac_vals  # skaliert nur für Sichtbarkeit
    plt.scatter(conf_vals, acc_vals, s=sizes)

    plt.xlabel("Heuristic confidence 1 - normalized_set_size")
    plt.ylabel("Empirical accuracy (Top-1 correct)")
    plt.title("Reliability diagram (ECE-style)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_bars(x, ys, labels, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(7, 4))
    for y, lab in zip(ys, labels):
        plt.plot(x, y, marker="o", label=lab)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(labels) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str)
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("qhat_json", type=str, help="path to saved q_hat JSON (from calibration)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=str, default="./work_dirs/cp_eval")
    ap.add_argument("--max-batches", type=int, default=-1, help="for debugging; -1 = full dataset")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.qhat_json) as f:
        meta = json.load(f)
    q_hat = float(meta["q_hat"])
    alpha = float(meta.get("alpha", 0.1))

    cfg = Config.fromfile(args.config)
    cfg.model.setdefault("test_cfg", {})
    cfg.model.test_cfg["output_logits"] = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg, args.checkpoint, device=device)

    dl_key = f"{args.split}_dataloader"
    ds_cfg = getattr(cfg, dl_key)["dataset"]
    dataset = build_dataset(ds_cfg)
    loader = build_loader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    # classes / palette
    if hasattr(dataset, "metainfo"):
        metainfo = dataset.metainfo
    elif hasattr(dataset, "METAINFO"):
        metainfo = dataset.METAINFO
    else:
        raise RuntimeError("Dataset has no metainfo/METAINFO with palette/classes")

    num_classes = len(metainfo["palette"])
    ignore_index = getattr(getattr(model, "decode_head", None), "ignore_index", 255)

    # ---- streaming counters per set size s = 1..C ----
    # counts_total[s], counts_err[s], counts_covered[s]
    counts_total = np.zeros(num_classes + 1, dtype=np.int64)
    counts_err = np.zeros(num_classes + 1, dtype=np.int64)
    counts_cov = np.zeros(num_classes + 1, dtype=np.int64)

    total_valid = 0
    total_err = 0
    total_cov = 0
    sum_set_size = 0

    # also keep distribution of uncertainty scores for quick sanity checks
    for i, batch in enumerate(loader):
        if args.max_batches >= 0 and i >= args.max_batches:
            break

        inputs = batch["inputs"]
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs, dim=0)
        inputs = inputs.to(device).contiguous()

        data_samples = batch["data_samples"]
        # move GT to device if present
        has_gt = True
        for ds in data_samples:
            if hasattr(ds, "gt_sem_seg") and ds.gt_sem_seg is not None:
                ds.gt_sem_seg.data = ds.gt_sem_seg.data.to(device, non_blocking=True)
            else:
                has_gt = False

        if not has_gt:
            raise RuntimeError(
                f"No ground-truth found in split='{args.split}'. "
                "Cityscapes official 'test' has no labels; use 'val' for evaluation."
            )

        pred_samples = model.predict(inputs, data_samples)

        logit_list = []
        for ds in pred_samples:
            if hasattr(ds, "seg_logits") and ds.seg_logits is not None:
                logit_list.append(ds.seg_logits.data)
            else:
                raise RuntimeError(
                    "predict() has no seg_logits. "
                    "Ensure cfg.model.test_cfg['output_logits'] = True"
                )

        logits = torch.stack(logit_list, dim=0)  # [B,C,H,W]
        probs = torch.softmax(logits, dim=1)

        # get gt labels aligned to model output shape via model preprocessor
        _, labels, _ = preprocess_batch(model, batch)  # labels: [B,H,W]
        if labels is None:
            raise RuntimeError("preprocess_batch() returned no labels; cannot evaluate.")

        valid = labels != ignore_index
        if not valid.any():
            continue

        # CP sets
        set_mask, set_size = build_sets_from_probs(probs, q_hat)  # set_size: [B,1,H,W], set_mask: [B,C,H,W] -> bool, True for classes in set
        set_size = set_size.squeeze(1)  # [B,H,W]

        # top-1 correctness (for error-detection evaluation)
        pred = probs.argmax(dim=1)  # [B,H,W]
        err = (pred != labels) & valid  # [B,H,W]
        if (labels[valid].max() >= num_classes) or (labels[valid].min() < 0):
            raise RuntimeError("GT labels out of range for num_classes")

        # CP coverage event: y in set
        gt_idx = labels.clone()
        gt_idx[~valid] = 0
        gt_idx = gt_idx.clamp(min=0, max=num_classes - 1).unsqueeze(1)  # [B,1,H,W]
        in_set = torch.gather(set_mask, dim=1, index=gt_idx).squeeze(1) & valid # pixels where GT class is in CP set

        # update totals
        vcount = int(valid.sum().item())
        ecount = int(err.sum().item())    # -> coming from GT
        ccount = int(in_set.sum().item()) # -> coming from CP

        total_valid += vcount # pixel-count over all batches
        total_err += ecount   # total error pixels
        total_cov += ccount   # total covered pixels
        sum_set_size += int(set_size[valid].sum().item())

        # per set size buckets
        ss_flat = set_size[valid].detach().cpu().numpy().astype(np.int64)
        err_flat = err[valid].detach().cpu().numpy().astype(np.int64)
        cov_flat = in_set[valid].detach().cpu().numpy().astype(np.int64)

        # vectorized bincount per bucket
        bc_total = np.bincount(ss_flat, minlength=num_classes + 1)                  # unique count of set sizes
        bc_err = np.bincount(ss_flat, weights=err_flat, minlength=num_classes + 1)  # error count per set size
        bc_cov = np.bincount(ss_flat, weights=cov_flat, minlength=num_classes + 1)  # coverage count per set size

        counts_total += bc_total.astype(np.int64)
        counts_err += bc_err.astype(np.int64)
        counts_cov += bc_cov.astype(np.int64)

        if (i + 1) % 10 == 0:
            print(f"[{i+1:04d}/{len(loader)}] valid_pix={total_valid:,}  cp_cov={safe_div(total_cov,total_valid):.4f}")

    # ---- metrics ----

    metrics = dict()

    cp_coverage = safe_div(total_cov, total_valid)
    mean_set_size = safe_div(sum_set_size, total_valid)
    top1_error_rate = safe_div(total_err, total_valid)


    # evaluate ECE for uncertainty calculation: 1 - ((set_size-1) / (num_classes-1))
    ece_bins = num_classes
    ece_style = 0.0
    ece_bin_acc = []
    ece_bin_conf = []
    ece_bin_frac = []
    ece_bin_s = []

    for s in range(1, num_classes +1):
        n_s = counts_total[s]
        if n_s == 0:
            continue
        
        err_s = counts_err[s]
        acc_s = 1 - (err_s / n_s)

        conf_s = 1.0 - ((s - 1.0) / (num_classes - 1.0)) # certainty score for this iteration -> in [0,1]
        frac_s = n_s / max(1, total_valid)

        ece_style += frac_s * abs(acc_s - conf_s)

        ece_bin_s.append(int(s))
        ece_bin_acc.append(float(acc_s))
        ece_bin_conf.append(float(conf_s))
        ece_bin_frac.append(float(frac_s))
    
    metrics |= {
        "ece_style_conf_from_setsize": float(ece_style),
        "ece_style_bins": {
            "set_size": ece_bin_s,
            "bin_fraction": ece_bin_frac,   
            "bin_accuracy": ece_bin_acc,
            "bin_confidence": ece_bin_conf, 
        }
    }


    # error-detection AUROC/AUPRC using uncertainty score = set_size (larger => more uncertain)
    # Build descending score bins: s = C..1
    pos_desc = []  # error pixels (positives)
    neg_desc = []  # correct pixels (negatives)
    for s in range(num_classes, 0, -1):
        t = counts_total[s]
        e = counts_err[s]
        pos_desc.append(e)
        neg_desc.append(t - e)

    fpr, tpr, prec, rec = roc_pr_from_bins(pos_desc, neg_desc)
    auroc = auc_trapz(fpr, tpr)
    auprc = auc_trapz(rec, prec)

    # Risk–Coverage: keep most certain pixels first (small set_size first)
    coverages = []
    risks = []
    kept = 0
    kept_err = 0
    for s in range(1, num_classes + 1):
        kept += counts_total[s] # number of pixels up until this set size
        kept_err += counts_err[s] # number of error pixels up until this set size
        coverages.append(safe_div(kept, total_valid))
        risks.append(safe_div(kept_err, kept))
    coverages = np.array(coverages, dtype=np.float64)
    risks = np.array(risks, dtype=np.float64)  # start at baseline risk

    aurc = aurc_from_risk_coverage(coverages, risks)

    metrics |= {
        "split": args.split,
        "num_classes": int(num_classes),
        "ignore_index": int(ignore_index),
        "q_hat": float(q_hat),
        "alpha": float(alpha),
        "target_coverage_(1-alpha)": float(1 - alpha),

        "total_valid_pixels": int(total_valid),
        "top1_error_rate": float(top1_error_rate),

        "cp_pixel_coverage_(y_in_set)": float(cp_coverage),
        "cp_mean_set_size": float(mean_set_size),

        "uncertainty_error_AUROC": float(auroc),
        "uncertainty_error_AUPRC": float(auprc),

        "risk_coverage_AURC": float(aurc),
    }

    out_json = osp.join(args.out, "metrics_cp_uncertainty.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {out_json}")
    print(json.dumps(metrics, indent=2))

    # ---- plots ----
    # 1) risk–coverage curve
    plot_risk_coverage(coverages, risks, osp.join(args.out, "risk_coverage_cp.png"))

    # 2) per set-size: error rate and CP-coverage rate
    s_vals = np.arange(1, num_classes + 1)
    err_rate_s = np.array([safe_div(counts_err[s], counts_total[s]) for s in s_vals], dtype=np.float64)
    cov_rate_s = np.array([safe_div(counts_cov[s], counts_total[s]) for s in s_vals], dtype=np.float64)
    plot_bars(
        s_vals, [err_rate_s, cov_rate_s], ["Top-1 error rate", "CP coverage rate (y in set)"],
        xlabel="Set size", ylabel="Rate", title="Error/Coverage vs. CP set size",
        out_path=osp.join(args.out, "error_coverage_by_setsize.png")
    )

    # 3) distribution of set sizes
    plt.figure(figsize=(7, 3.5))
    frac = counts_total[1:] / max(1, counts_total[1:].sum())
    plt.bar(np.arange(1, num_classes + 1), frac)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Set size")
    plt.ylabel("Fraction of valid pixels")
    plt.title("Distribution of CP set sizes")
    plt.tight_layout()
    plt.savefig(osp.join(args.out, "setsize_distribution.png"), dpi=200)
    plt.close()

    # 4) ROC Curve + AUROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"CP (AUROC = {auroc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Error Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(osp.join(args.out, "roc_cp_uncertainty.png"), dpi=200)
    plt.close()

    #5) PR Curve + AUPRC
    plt.figure()
    plt.plot(rec, prec, label=f"CP (AUPRC = {auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve – Error Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(osp.join(args.out, "pr_cp_uncertainty.png"), dpi=200)
    plt.close()

    # 6) Reliability diagram (ECE-style)
    plot_reliability_discrete(
        metrics["ece_style_bins"]["bin_confidence"],
        metrics["ece_style_bins"]["bin_accuracy"],
        metrics["ece_style_bins"]["bin_fraction"],
        osp.join(args.out, "reliability_ece_style.png"),
    )

    print(f"Saved plots to: {args.out}")


if __name__ == "__main__":
    main()
