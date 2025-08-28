import os.path as osp
import os, json, csv
import numpy as np
from PIL import Image
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import pdb

@HOOKS.register_module()
class UncertaintyDumpHook(Hook):
    def __init__(self, out_dir=None, save_maps=True):
        self.out_dir = out_dir  # if None -> work_dir/uncertainty_out
        self.save_maps = save_maps
        self.rows = []

    def before_test(self, runner):
        base = self.out_dir or osp.join(runner.work_dir, 'uncertainty_out')
        self.out_dir = base
        self.img_dir = osp.join(base, 'uncertainty_maps')
        os.makedirs(self.out_dir, exist_ok=True)
        if self.save_maps:
            os.makedirs(self.img_dir, exist_ok=True)
        self.rows.clear()

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not outputs:
            return
        for sample in outputs:
            meta = getattr(sample, 'metainfo', {})
            base = meta.get('ori_filename') or meta.get('img_path') or f'img_{len(self.rows):06d}'
            base = osp.splitext(osp.basename(str(base)))[0]

            score = getattr(sample, 'uncertainty_score', None)
            if score is not None:
                self.rows.append({'image': base, 'uncertainty': float(score)})

            if self.save_maps:
                uc = getattr(sample, 'uncertainty_map', None)
                if uc is not None and hasattr(uc, 'data'):
                    # visualize
                    m = uc.data.squeeze(0).float().cpu().numpy()
                    m = m - m.min()  # min to 0
                    denom = (m.max() + 1e-12) # dont divide by 0
                    arr = (m / denom * 255.0).astype(np.uint8)
                    Image.fromarray(arr).save(osp.join(self.img_dir, f'{base}_uncertainty.png'))

    def after_test(self, runner):
        csv_path = osp.join(self.out_dir, 'uncertainty_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['image', 'uncertainty'])
            w.writeheader()
            w.writerows(self.rows)
        json_path = osp.join(self.out_dir, 'uncertainty_summary.json')
        with open(json_path, 'w') as f:
            json.dump(self.rows, f, indent=2)

        vals = [r['uncertainty'] for r in self.rows if 'uncertainty' in r]
        if vals:
            mean_u = float(np.mean(vals)); median_u = float(np.median(vals))
            runner.logger.info(
                f"[UncertaintyDumpHook] wrote {len(self.rows)} rows | "
                f"mean={mean_u:.4f}, median={median_u:.4f} -> {csv_path}"
            )
            
            runner.message_hub.update_scalar('uncertainty/mean', mean_u)
            runner.message_hub.update_scalar('uncertainty/median', median_u)