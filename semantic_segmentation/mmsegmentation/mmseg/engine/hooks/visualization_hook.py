# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample

import torch
import torch.nn.functional as F

@torch.no_grad()
def miou_from_seg_data_sample(sample, num_classes: int = 19, ignore_index: int = 255):
    """Berechnet mIoU für genau ein SegDataSample."""
    # 1) Ground truth (H, W) – in MMSeg ist das meist (1, H, W)
    gt = sample.gt_sem_seg.data.cpu()
    if gt.dim() == 3 and gt.size(0) == 1:
        gt = gt.squeeze(0)
    gt = gt.to(torch.long)

    # 2) Vorhersage-Logits (C, H, W) -> Labelmap (H, W)
    logits = sample.seg_logits.data.cpu()  # (C, H, W)
    # Falls Größen nicht matchen, Logits auf GT-Größe bringen
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits.unsqueeze(0), size=gt.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    pred = logits.argmax(dim=0).to(torch.long)

    # 3) Ignore-Maske
    mask = gt != ignore_index
    if mask.sum() == 0:
        return {"miou": float("nan"), "per_class_iou": torch.full((num_classes,), float("nan"))}

    gt_v = gt[mask]
    pred_v = pred[mask]

    # 4) Konfusionsmatrix
    hist = torch.bincount(
        (gt_v * num_classes + pred_v).to(torch.long),
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).to(torch.float64)

    # 5) IoU pro Klasse und mIoU
    diag = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - diag
    valid = union > 0
    iou = torch.zeros(num_classes, dtype=torch.float64)
    iou[valid] = (diag[valid] / union[valid])
    miou = iou[valid].mean().item() if valid.any() else float("nan")

    return {"miou": float(miou), "per_class_iou": iou.double()}


@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50, #1,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return

        # import pdb
        # print(outputs[0].seg_map_path)
        # print(miou_from_seg_data_sample(outputs[0]))
        # pdb.set_trace()

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)

