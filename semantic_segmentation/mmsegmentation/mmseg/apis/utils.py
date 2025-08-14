# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Sequence, Union
from copy import deepcopy
import numpy as np
from mmengine.dataset import Compose
from mmengine.model import BaseModel
import os

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def get_cityscapes_ann(img_path):
    new_path = img_path.replace(os.sep + "leftImg8bit " + os.sep, os.sep + "gtFine" + os.sep)
    stem, filetype = os.path.splitext(new_path)
    if stem.endswith("_leftImg8bit"):
        stem = stem.split("_leftImg8bit")[0]
    ann_path = base + "gtFine_labelTrainIds.png"
    return ann_path

def _prepare_data_with_ann(imgs: ImageType,
                           model: BaseModel,
                           anns: ImageType | list[str] | None = None,
                           derive_ann_fn=None):
    """
    anns: optional, if None -> derive_ann_fn
    derive_ann_fn: if None -> cityscapes
    """
    cfg = model.cfg
    test_pipeline_copy = deepcopy(cfg.test_pipeline)

    has_load_ann = any(t.get('type') == 'LoadAnnotations' for t in test_pipeline)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    test_pipeline = deepcopy(cfg.test_pipeline)

    pipeline = Compose(test_pipeline)

    data = defaultdict(list)

    for i, img in enumerate(imgs):
        item = {}
        img_is_array = isinstance(img, np.ndarray)

        if img_is_array:
            item['img'] = img
            if has_load_ann:
                if anns is not None:
                    seg_path = anns[i]
                    item['seg_map_path'] = seg_path
                else:
                    raise ValueError("LoadAnnotations only works with paths as imgs")
        else:
            item['img_path'] = img
            if has_load_ann:
                seg_path = None
                if anns is not None:
                    seg_path = anns[i]
                elif derive_ann_fn is not None:
                    seg_path = derive_ann_fn(img)
                else:
                    seg_path = derive_cityscapes_ann(img)
                item['seg_map_path'] = seg_path

        out = pipeline(item)
        data['inputs'].append(out['inputs'])
        data['data_samples'].append(out['data_samples'])

    return data, is_batch

def _preprare_data(imgs: ImageType, model: BaseModel):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t) 

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch
