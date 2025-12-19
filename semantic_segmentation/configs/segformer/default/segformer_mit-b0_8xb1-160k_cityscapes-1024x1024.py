import os
_base_ = [
    os.path.join(os.environ["MMSEG_CONFIGS"], 'segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py')
]

model = dict(
    enable_normalization = True,
    normalize_mean_std=dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    perform_attack = False, # False # attacks while testing
    adv_train_enable = False,      # attacks while training
    mc_dropout = False,
    mc_runs = 8,
    adv_train_ratio = 0.5,
    decode_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, reduction="none" # added reduction='none', if error on loss -> line before `loss.backward()`, add loss=loss.mean()
            # type='EvidentialMSELoss', loss_weight = 1.0, kl_strength = 1.0, reduction = "mean"
            )
    ),
    attack_loss = dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, reduction="none"
        # type='EvidentialMSELoss', loss_weight = 1.0, kl_strength = 1.0, reduction = "none"
    ),
    attack_cfg = {"name": "cospgd", "norm": "linf","epsilon": 4,"alpha": 0.01, "iterations": 20, "targeted": False}
    # attack_cfg = {"name": "pgd", "norm": "l2","epsilon": 8,"alpha": 0.01, "iterations": 20}
)

custom_imports = dict(
    imports=['mmseg_custom.hooks.uncertainty_dump'],
    allow_failed_imports=False
)
custom_hooks = [
    dict(
        type='UncertaintyDumpHook',
        save_maps=True
    )
]

train_dataloader = dict(batch_size=1, num_workers=8, pin_memory=True, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=8)
test_dataloader = val_dataloader