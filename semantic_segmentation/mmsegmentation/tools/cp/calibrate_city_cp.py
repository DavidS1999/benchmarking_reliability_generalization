import os, os.path as osp, json, math, argparse
import torch
from typing import List

IGNORE_INDEX = 255

# mmengine/mmseg (>=1.x)
from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.registry import DATASETS
from torch.utils.data import DataLoader, Subset
from mmengine.dataset import pseudo_collate
from cp_utils import preprocess_batch, aps_scores_for_batch, cp_quantile

import pdb

# ========= Utils: City-Parsing =========
def city_from_img_path(p: str) -> str:
    """
    Cityscapes Struktur:
      .../leftImg8bit/{split}/{CITY}/{CITY}_{seq}_{frame}_leftImg8bit.png
    Wir nehmen primär den Ordnernamen (CITY), fallback Dateiprfix.
    """
    d = osp.dirname(p)
    city = osp.basename(d)
    if not city or city in {"train", "val", "test"}:
        city = osp.basename(p).split("_")[0]
    return city


def get_any_img_path(info: dict) -> str:
    """Robust einen Pfad aus einem data_info holen (serialisiert/verschiedene Keys)."""
    if "img_path" in info and info["img_path"]:
        return info["img_path"]
    if "seg_map_path" in info and info["seg_map_path"]:
        return info["seg_map_path"]
    if "img_info" in info and isinstance(info["img_info"], dict) and "filename" in info["img_info"]:
        return info["img_info"]["filename"]
    if "filename" in info:
        return info["filename"]
    raise KeyError("Konnte Bildpfad in data_info nicht finden.")


# ========= Dataset/Dataloader (mmengine) =========
def build_dataset_mmengine(ds_cfg: dict):
    """Baut und initialisiert das Dataset (full_init), damit get_data_info nutzbar ist."""
    dataset = DATASETS.build(ds_cfg)
    if hasattr(dataset, "full_init"):
        dataset.full_init()  # füllt interne Strukturen; kann data_list serialisieren
    return dataset


def build_dataloader_simple(dataset, batch_size: int = 2, num_workers: int = 2, shuffle: bool = False):
    """
    Einfacher DataLoader. Kein spezieller Sampler nötig; Reihenfolge egal.
    Wichtig: pseudo_collate nutzen, damit SegDataSample korrekt collated.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pseudo_collate,
        pin_memory=True,
        drop_last=False,
    )


# ========= Indizes für eine Stadt bestimmen (serialisiert-sicher) =========
def find_city_indices(dataset, city_name: str) -> List[int]:
    """
    Baut eine Indexliste aller Beispiele, deren (Bild- oder Label-)Pfad zu 'city_name' gehört.
    Nutzt get_data_info(i), funktioniert auch wenn data_list serialisiert wurde.
    """
    idxs = []
    n = len(dataset)
    for i in range(n):
        info = dataset.get_data_info(i)  # liest serialisierte Einträge korrekt
        p = get_any_img_path(info)
        if city_from_img_path(p) == city_name:
            idxs.append(i)

    # Fallback-Heuristik: falls oben nichts gefunden, suche über alle Pfadteile/Dateinamen
    if not idxs:
        for i in range(n):
            info = dataset.get_data_info(i)
            p = get_any_img_path(info)
            parts = osp.normpath(p).split(os.sep)
            if city_name in parts or osp.basename(p).startswith(city_name + "_"):
                idxs.append(i)

    if not idxs:
        # Debug-Hinweis mit Beispielpfad
        example = get_any_img_path(dataset.get_data_info(0))
        raise ValueError(
            f"Keine Dateien für Stadt '{city_name}' gefunden.\n"
            f"Beispielpfad aus Dataset: {example}\n"
            f"Prüfe Split/Config und exakte Schreibweise der Stadt."
        )
    return idxs



# ========= Kalibrierungsschleife =========
@torch.no_grad()
def calibrate_city(model,
                   dataloader,
                   device: str = "cuda",
                   alpha: float = 0.1,
                   randomized: bool = False):
    model.eval()
    all_scores = []
    for batch in dataloader:
        imgs, labels, img_metas = preprocess_batch(model, batch)
        if labels is None:
            raise RuntimeError("No labels in Batch – use split with GT (train/val).")
        
        logits = model.encode_decode(imgs, img_metas)  # [B,C,H,W]
        scores = aps_scores_for_batch(logits, labels, randomized=randomized)
        if scores.numel() > 0:
            all_scores.append(scores.cpu())

    if not all_scores:
        raise ValueError("Es wurden keine gültigen Pixel-Scores gesammelt (prüfe ignore_index & Labels).")
    
    scores = torch.cat(all_scores, dim=0)
    pdb.set_trace()
    q_hat = cp_quantile(scores, alpha)
    return q_hat, scores.numel()


# ========= Main =========
def main():
    parser = argparse.ArgumentParser(description="Conformal Calibration (APS) pro Stadt (Cityscapes, mmseg 1.2.2)")
    parser.add_argument("config", type=str, help="Pfad zur mmseg Config (.py)")
    parser.add_argument("checkpoint", type=str, help="Pfad zum Checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Cityscapes-Split für Kalibrierung")
    parser.add_argument("--city", type=str, required=True, help="Stadtname (z.B. Frankfurt, Lindau, ...)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Signifikanzniveau (z.B. 0.1 für 90%% Abdeckung)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--randomized", action="store_true", help="Randomized APS statt deterministisch")
    parser.add_argument("--work-dir", type=str, default="./work_dirs/cp_calibration")
    args = parser.parse_args()

    # Config laden
    cfg = Config.fromfile(args.config)

    # Batchsize/Workers im gewünschten Split setzen
    dl_key = f"{args.split}_dataloader"
    if hasattr(cfg, dl_key):
        getattr(cfg, dl_key)["batch_size"] = args.batch_size
        getattr(cfg, dl_key)["num_workers"] = args.workers
    else:
        raise KeyError(f"Erwarte '{dl_key}' in der Config.")

    # Modell laden
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg, args.checkpoint, device=device)

    # Dataset bauen
    if "dataset" not in getattr(cfg, dl_key):
        raise KeyError(f"Erwarte '{dl_key}[\"dataset\"]' in der Config.")
    ds_cfg = getattr(cfg, dl_key)["dataset"]
    dataset = build_dataset_mmengine(ds_cfg)

    # City-Indizes bestimmen und Subset bilden
    city_idxs = find_city_indices(dataset, args.city)
    subset = Subset(dataset, city_idxs)

    # Debug (optional):
    info0 = dataset.get_data_info(city_idxs[0])
    print(f"Erstes Sample der Stadt '{args.city}':", get_any_img_path(info0))
    print(f"Samples in Stadt-Subset: {len(subset)}")

    # Dataloader
    dataloader = build_dataloader_simple(
        subset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False
    )

    # Kalibrierung
    q_hat, n_pix = calibrate_city(model, dataloader, device=device,
                                  alpha=args.alpha, randomized=args.randomized)

    # Speichern
    os.makedirs(args.work_dir, exist_ok=True)
    out_path = osp.join(args.work_dir, f"cp_qhat_city-{args.city}_{args.split}_alpha-{args.alpha}.json")
    with open(out_path, "w") as f:
        json.dump({
            "alpha": args.alpha,
            "city": args.city,
            "split": args.split,
            "randomized": bool(args.randomized),
            "q_hat": q_hat,
            "n_pixels": int(n_pix)
        }, f, indent=2)

    print(f"[OK] q_hat = {q_hat:.6f} (n_pixels={n_pix}) → gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()