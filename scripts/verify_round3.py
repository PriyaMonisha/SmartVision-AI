#!/usr/bin/env python3
"""
scripts/verify_round3.py
Pre-Colab verification for SmartVision Round 3.
Run: python scripts/verify_round3.py
Exit 0 = all pass. Exit 1 = failures.

Checks:
  1. Augmentor: 10 transforms, PIL-before-tensor order, no RandomErasing,
     eval pipeline deterministic.
  2. MobileNetV2 Phase 1: Dropout(0.4), features frozen, ~28K head params.
  3. MobileNetV2 Phase 2: features[0:15] frozen, features[16:] unfrozen,
     param count in [900K, 1.6M].
  4. BN freeze: _freeze_backbone_bn correctness after partial unfreeze.
  5. ResNet50: Phase 1 (only fc.* trainable), BUG-1 (fc unfrozen),
     layer4.0/4.1 still frozen, Phase 2 count in [3M, 7M].
"""

import sys
import importlib

import torch
import torch.nn as nn

sys.path.insert(0, ".")

_failures: list[str] = []


def check(condition: bool, pass_msg: str, fail_msg: str) -> bool:
    if condition:
        print(f"    PASS  {pass_msg}")
    else:
        print(f"    FAIL  {fail_msg}")
        _failures.append(fail_msg)
    return condition


# ==========================================================================
# CHECK 1: Augmentor pipeline
# ==========================================================================
print("=" * 65)
print("CHECK 1: Augmentor pipeline")
print("=" * 65)

import src.data.augmentor as _aug_mod
importlib.reload(_aug_mod)
from src.data.augmentor import get_train_transforms, get_eval_transforms, IMAGENET_MEAN

train_pipe = get_train_transforms(224)
names      = [type(t).__name__ for t in train_pipe.transforms]
idx        = {n: i for i, n in enumerate(names)}
print(f"  Pipeline ({len(names)}): {names}")

check(len(names) == 10, "10 transforms", f"Expected 10, got {len(names)}: {names}")

for req in ["RandomHorizontalFlip", "RandomRotation", "ColorJitter", "RandomGrayscale",
            "RandomPerspective", "ToImage", "ToDtype", "RandomZoomOut", "Resize", "Normalize"]:
    check(req in names, f"{req} present", f"{req} MISSING")

check("RandomErasing" not in names, "RandomErasing absent",
      "RandomErasing PRESENT -- must be removed")

if "RandomGrayscale" in idx and "ToImage" in idx:
    check(idx["RandomGrayscale"] < idx["ToImage"],
          f"RandomGrayscale[{idx['RandomGrayscale']}] before ToImage[{idx['ToImage']}] (PIL-space)",
          "RandomGrayscale must precede ToImage")

if "RandomPerspective" in idx and "ToImage" in idx:
    check(idx["RandomPerspective"] < idx["ToImage"],
          f"RandomPerspective[{idx['RandomPerspective']}] before ToImage[{idx['ToImage']}] (PIL-space)",
          "RandomPerspective must precede ToImage")

if "ToDtype" in idx and "RandomZoomOut" in idx:
    check(idx["ToDtype"] < idx["RandomZoomOut"],
          f"ToDtype[{idx['ToDtype']}] before RandomZoomOut[{idx['RandomZoomOut']}]",
          "ToDtype must precede RandomZoomOut (float32 for fill)")

if "RandomZoomOut" in idx and "Normalize" in idx:
    check(idx["RandomZoomOut"] < idx["Normalize"],
          f"RandomZoomOut[{idx['RandomZoomOut']}] before Normalize[{idx['Normalize']}]",
          "RandomZoomOut must precede Normalize")

_stochastic = {"RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation",
               "ColorJitter", "RandomGrayscale", "RandomPerspective",
               "RandomZoomOut", "RandomErasing", "RandomCrop", "RandomResizedCrop",
               "RandomAffine", "RandomAutocontrast", "RandomEqualize"}
_eval_names = [type(t).__name__ for t in get_eval_transforms(224).transforms]
_found_s    = set(_eval_names) & _stochastic
check(not _found_s, f"Eval deterministic: {_eval_names}",
      f"Eval contains stochastic transforms: {_found_s}")

# ==========================================================================
# CHECK 2: MobileNetV2 Phase 1
# ==========================================================================
print("\n" + "=" * 65)
print("CHECK 2: MobileNetV2 Phase 1")
print("=" * 65)

from src.models.model_factory import get_model

model = get_model("mobilenet", num_classes=22)
_c0, _c1 = model.classifier[0], model.classifier[1]

check(isinstance(_c0, nn.Dropout), "classifier[0] is Dropout",
      f"classifier[0] is {type(_c0).__name__}, expected Dropout")
if isinstance(_c0, nn.Dropout):
    check(abs(_c0.p - 0.4) < 1e-6, "Dropout p=0.40",
          f"Dropout p={_c0.p:.4f}, expected 0.40")

check(isinstance(_c1, nn.Linear), "classifier[1] is Linear",
      f"classifier[1] is {type(_c1).__name__}")
if isinstance(_c1, nn.Linear):
    check(_c1.in_features == 1280 and _c1.out_features == 22,
          "Linear(1280, 22)", f"Linear({_c1.in_features}, {_c1.out_features})")

_feat_train = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
check(_feat_train == 0, "All features frozen",
      f"{_feat_train:,} features params trainable -- freeze failed")

_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
check(20_000 < _p1 < 50_000, f"Phase 1 trainable={_p1:,} in [20K,50K]",
      f"Phase 1 trainable={_p1:,} outside [20K,50K]. Expected ~28K.")

# ==========================================================================
# CHECK 3: MobileNetV2 Phase 2
# ==========================================================================
print("\n" + "=" * 65)
print("CHECK 3: MobileNetV2 Phase 2 unfreeze")
print("=" * 65)

from src.models.model_factory import unfreeze_mobilenet_phase2
unfreeze_mobilenet_phase2(model)

_frz_wrong   = [i for i, b in enumerate(model.features)
                if i < 16 and any(p.requires_grad for p in b.parameters())]
_unfrz_wrong = [i for i, b in enumerate(model.features)
                if i >= 16 and not any(p.requires_grad for p in b.parameters())]

check(len(_frz_wrong) == 0, "features[0:15] all frozen",
      f"features{_frz_wrong} should be frozen but have trainable params")
check(len(_unfrz_wrong) == 0, "features[16:18] all unfrozen",
      f"features{_unfrz_wrong} should be unfrozen but have no trainable params")

_p2  = sum(p.numel() for p in model.parameters() if p.requires_grad)
_ppi = _p2 / 3080
print(f"  Phase 2 trainable: {_p2:,} ({_ppi:.0f}/img)")
check(900_000 < _p2 < 1_600_000, f"Param count {_p2:,} in [900K, 1.6M]",
      f"Param count {_p2:,} outside [900K, 1.6M]. Expected ~1.23M.")
check(200 < _ppi < 600, f"Params/img={_ppi:.0f} in [200, 600]",
      f"Params/img={_ppi:.0f} outside [200, 600]")

# ==========================================================================
# CHECK 4: BN freeze behavior
# ==========================================================================
print("\n" + "=" * 65)
print("CHECK 4: BatchNorm freeze (_freeze_backbone_bn)")
print("=" * 65)

from src.models.base_classifier import _freeze_backbone_bn

_bn_model = get_model("mobilenet", num_classes=22)
unfreeze_mobilenet_phase2(_bn_model)
_bn_model.train()
_freeze_backbone_bn(_bn_model)

_bad_frz  = [(i, type(m).__name__) for i, b in enumerate(_bn_model.features)
             for m in b.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
             if i < 16 and m.training]
_bad_unfrz = [(i, type(m).__name__) for i, b in enumerate(_bn_model.features)
              for m in b.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
              if i >= 16 and not m.training]

check(len(_bad_frz) == 0, "features[0:15] BN in eval() (stats locked)",
      f"{len(_bad_frz)} BN in frozen blocks still in train(): {_bad_frz[:3]}")
check(len(_bad_unfrz) == 0, "features[16:] BN in train() (stats updating)",
      f"{len(_bad_unfrz)} BN in unfrozen blocks in eval(): {_bad_unfrz}")
check(_bn_model.classifier.training, "classifier in train()",
      "classifier in eval() -- should be training")

# ==========================================================================
# CHECK 5: ResNet50
# ==========================================================================
print("\n" + "=" * 65)
print("CHECK 5: ResNet50 (Phase 1 + BUG-1 regression + scope)")
print("=" * 65)

from src.models.model_factory import unfreeze_resnet50_phase2

rn = get_model("resnet50", num_classes=22)

# Phase 1: only fc.* trainable
_rn_p1  = sum(p.numel() for p in rn.parameters() if p.requires_grad)
_rn_fc1 = sum(p.numel() for n, p in rn.named_parameters()
              if p.requires_grad and n.startswith("fc."))
check(_rn_p1 > 0 and _rn_p1 == _rn_fc1,
      f"Phase 1: only fc.* trainable ({_rn_fc1:,})",
      f"Phase 1: total={_rn_p1:,} fc={_rn_fc1:,}. Expected total==fc.")

unfreeze_resnet50_phase2(rn)

_fc = sum(p.numel() for n, p in rn.named_parameters()
          if p.requires_grad and n.startswith("fc."))
check(_fc > 0, f"fc unfrozen: {_fc:,}",
      "fc NOT unfrozen. BUG-1 REGRESSION. Check name.startswith('fc.')")

_l42 = sum(p.numel() for n, p in rn.named_parameters()
           if p.requires_grad and n.startswith("layer4.2"))
check(_l42 > 0, f"layer4.2 unfrozen: {_l42:,}",
      "layer4.2 NOT unfrozen. Check prefix == 'layer4.2'")

check(not any(p.requires_grad for n, p in rn.named_parameters() if n.startswith("layer4.0")),
      "layer4.0 frozen", "layer4.0 trainable -- scope too broad")
check(not any(p.requires_grad for n, p in rn.named_parameters() if n.startswith("layer4.1")),
      "layer4.1 frozen", "layer4.1 trainable -- scope too broad")

_rn_p2 = sum(p.numel() for p in rn.parameters() if p.requires_grad)
print(f"  ResNet50 Phase 2: {_rn_p2:,} ({_rn_p2/3080:.0f}/img)")
check(3_000_000 < _rn_p2 < 7_000_000,
      f"Phase 2 count {_rn_p2:,} in [3M, 7M]",
      f"Phase 2 count {_rn_p2:,} outside [3M, 7M]")

# ==========================================================================
# Final
# ==========================================================================
print("\n" + "=" * 65)
if not _failures:
    print("ALL CHECKS PASSED -- Ready for Colab Round 3")
    print("=" * 65)
    sys.exit(0)
else:
    print(f"FAILED -- {len(_failures)} check(s):")
    for i, f in enumerate(_failures, 1):
        print(f"  {i:2d}. {f}")
    print("=" * 65)
    sys.exit(1)
