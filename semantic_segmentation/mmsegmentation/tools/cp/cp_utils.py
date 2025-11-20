import math, torch
from mmseg.apis import init_model
from mmengine.config import Config

IGNORE_INDEX = 255

@torch.no_grad()
def preprocess_batch(model, batch):
    packed = {"inputs": batch["inputs"], "data_samples": batch["data_samples"]}
    out = model.data_preprocessor(packed, training=False)
    imgs = out["inputs"]                            # [B,3,H,W], float32
    samples = out["data_samples"]                   # List[SegDataSample]
    labels = None
    if hasattr(samples[0], "gt_sem_seg") and samples[0].gt_sem_seg is not None:
        labels = torch.stack([ds.gt_sem_seg.data for ds in samples], dim=0).squeeze(1)  # [B,H,W]
    img_metas = [ds.metainfo for ds in samples]
    return imgs, labels, img_metas

@torch.no_grad()
def aps_scores_for_batch(logits: torch.Tensor, labels: torch.Tensor, randomized: bool = False):
    """APS-Score per pixel."""
    import pdb
    pdb.set_trace()
    B, C, H, W = logits.shape

    probs = torch.softmax(logits, dim=1).permute(0,2,3,1).reshape(-1, C)

    y = labels.reshape(-1)

    valid = y != IGNORE_INDEX

    probs = probs[valid]
    y = y[valid]

    if probs.numel() == 0:
        return probs.new_empty(0)
    
    sorted_probs, sorted_idx = probs.sort(dim=1, descending=True)

    r = (sorted_idx == y.unsqueeze(1)).float().argmax(dim=1) # position of correct class in sorted probs

    cumsums = sorted_probs.cumsum(dim=1) # i.e. multiple rows of [0.9994, 0.9996, 0.9998, 0.9999, 1.0000]

    base = torch.zeros_like(r, dtype=sorted_probs.dtype)

    has_prefix = r > 0 # r=0 -> highest prob is correct class, r = 1 -> second highest prob is correct class, ...

    base[has_prefix] = cumsums[has_prefix, r[has_prefix]-1] # list of cumsum before correct class -> sum of probs before correct class

    on_rank = sorted_probs[torch.arange(sorted_probs.size(0), device=probs.device), r] # prob of correct class

    U = torch.rand_like(on_rank) if randomized else torch.ones_like(on_rank)

    return base + U * on_rank

def cp_quantile(scores_1d: torch.Tensor, alpha: float) -> float:
    """split-CP quantile with (n+1) and ceil."""
    n = scores_1d.numel()
    if n == 0:
        raise ValueError("Keine gültigen Pixel zum Kalibrieren gefunden.")
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = max(1, min(k, n))
    return float(scores_1d.kthvalue(k).values.item())

@torch.no_grad()
def build_sets_from_probs(probs: torch.Tensor, q_hat, top1_proxy_mondrian=False):
    """
    probs:  [B,C,H,W]  (Softmax)
    q_hat:  float ODER Tensor[C] (Mondrian-Schwellen)
    returns:
      set_mask: [B,C,H,W] bool
      set_size: [B,1,H,W] int
    """
    B, C, H, W = probs.shape

    x = probs.view(B, C, -1).permute(0, 2, 1).contiguous()           # [B,HW,C]
    sorted_probs, sorted_idx = x.sort(dim=2, descending=True)        # [B,HW,C]
    cumsums = sorted_probs.cumsum(dim=2)                              # [B,HW,C]


    if isinstance(q_hat, torch.Tensor) and q_hat.numel() == C and top1_proxy_mondrian:
        top1_idx = probs.argmax(dim=1).view(B, -1)                    # [B,HW]
        pixel_q = q_hat.to(probs.device)[top1_idx].unsqueeze(-1)      # [B,HW,1]
        thresh_mask = (cumsums >= pixel_q)
    else:
        thresh_mask = (cumsums >= float(q_hat))


    m = thresh_mask.float().argmax(dim=2)                             # [B,HW]

    arangeC = torch.arange(C, device=probs.device).view(1, 1, C)
    set_mask_sorted = (arangeC <= m.unsqueeze(-1)).bool()             # [B,HW,C]

    inv = sorted_idx.argsort(dim=2)                                   # [B,HW,C]

    set_mask_flat = set_mask_sorted.gather(2, inv).contiguous()       # [B,HW,C]
    set_mask = set_mask_flat.permute(0, 2, 1).view(B, C, H, W)        # [B,C,H,W]
    set_size = set_mask.sum(dim=1, keepdim=True)                      # [B,1,H,W]
    return set_mask, set_size
