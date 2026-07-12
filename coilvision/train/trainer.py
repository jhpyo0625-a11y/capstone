"""Two-stage fine-tune loop, early stopping, checkpoints, config snapshots (spec §6.3).

Stage 1: classifier head only (lr 3e-3, 3 epochs). Stage 2: full network
(lr 1e-4, cosine decay, up to 30 epochs), early stop on val (fail-recall,
macro-F1) with patience. Loss: class-weighted CE, weights from train split only.

One command, reproducible (fixed seed, config snapshot saved per run):
    uv run python -m coilvision.train.trainer [--smoke]

Outputs:
    models/<run_id>/model.pt                     best checkpoint + metadata
    artifacts/runs/<run_id>/config_snapshot.yaml
    artifacts/runs/<run_id>/history.csv          per-epoch metrics
    artifacts/runs/<run_id>/summary.json         best metrics, wall time
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from torch import nn

from coilvision.config import load_config, resolve_path
from coilvision.data.preprocess import preprocess_fingerprint
from coilvision.eval.metrics import fail_auc, fail_recall, false_reject_rate, macro_f1
from coilvision.train.datamodule import make_loaders


class EarlyStopper:
    """Tracks the best (fail_auc, macro_f1) tuple; stops after `patience` non-improving epochs.

    NOT fail-recall: selecting on recall alone picks the degenerate all-fail
    predictor (recall 1.0 at false-reject 1.0 — observed on run 20260711_231842).
    AUC is threshold-free; the operating threshold is tuned later (Phase 4).
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best: tuple | None = None
        self.best_epoch = -1
        self.bad = 0

    def update(self, key: tuple, epoch: int) -> bool:
        if self.best is None or key > self.best:
            self.best, self.best_epoch, self.bad = key, epoch, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader, fail_indices: list[int]) -> dict:
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            all_probs.append(torch.softmax(model(x), dim=1).numpy())
            all_y.append(y.numpy())
    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_y)
    y_pred = probs.argmax(axis=1)
    return {
        "fail_auc": fail_auc(y_true, probs, fail_indices),
        "fail_recall": fail_recall(y_true, probs, fail_indices),
        "macro_f1": macro_f1(y_true, y_pred),
        "false_reject_rate": false_reject_rate(y_true, probs, fail_indices),
        "accuracy": float((y_pred == y_true).mean()),
    }


def train_one_epoch(model: nn.Module, loader, criterion, optimizer) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(y)
        n += len(y)
    return total / max(n, 1)


def train_run(cfg: dict, smoke: bool = False) -> str:
    t0 = time.time()
    tc = cfg["train"]
    classes = cfg["data"]["classes"]
    fail_indices = [i for i, c in enumerate(classes) if c != "Pass"]
    run_id = time.strftime("%Y%m%d_%H%M%S") + ("_smoke" if smoke else "")
    seed_everything(tc["seed"])

    limits = (48, 24) if smoke else (None, None)
    train_loader, val_loader, weights = make_loaders(cfg, limit_train=limits[0], limit_val=limits[1])
    print(f"run {run_id}: train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"class_weights={[round(float(w), 3) for w in weights]}")

    model = timm.create_model(tc["backbone"], pretrained=True, num_classes=len(classes))
    criterion = nn.CrossEntropyLoss(weight=weights)
    history: list[dict] = []

    # Stage 1: head only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=tc["stage1"]["lr"])
    s1_epochs = 1 if smoke else tc["stage1"]["epochs"]
    for epoch in range(s1_epochs):
        loss = train_one_epoch(model, train_loader, criterion, opt)
        m = evaluate(model, val_loader, fail_indices)
        history.append({"stage": 1, "epoch": epoch, "train_loss": round(loss, 4), **{k: round(v, 4) for k, v in m.items()}})
        print(f"  s1 e{epoch}: loss={loss:.4f} val_fail_auc={m['fail_auc']:.3f} macro_f1={m['macro_f1']:.3f}")

    # Stage 2: full network, cosine decay, early stopping
    for p in model.parameters():
        p.requires_grad = True
    s2_epochs = 1 if smoke else tc["stage2"]["max_epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["stage2"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s2_epochs)
    stopper = EarlyStopper(tc["early_stop_patience"])
    best_state, best_metrics = copy.deepcopy(model.state_dict()), None
    for epoch in range(s2_epochs):
        loss = train_one_epoch(model, train_loader, criterion, opt)
        sched.step()
        m = evaluate(model, val_loader, fail_indices)
        history.append({"stage": 2, "epoch": epoch, "train_loss": round(loss, 4), **{k: round(v, 4) for k, v in m.items()}})
        improved = stopper.update((m["fail_auc"], m["macro_f1"]), epoch)
        if improved:
            best_state, best_metrics = copy.deepcopy(model.state_dict()), m
        print(f"  s2 e{epoch}: loss={loss:.4f} val_fail_auc={m['fail_auc']:.3f} "
              f"macro_f1={m['macro_f1']:.3f} recall={m['fail_recall']:.3f} "
              f"frr={m['false_reject_rate']:.3f}{' *best*' if improved else ''}")
        if stopper.should_stop:
            print(f"  early stop at epoch {epoch} (best epoch {stopper.best_epoch})")
            break
    if best_metrics is None:  # stage 2 never improved on init; evaluate the initial state
        best_metrics = evaluate(model, val_loader, fail_indices)

    wall_min = (time.time() - t0) / 60

    model_dir = resolve_path(cfg, "models_dir") / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "backbone": tc["backbone"],
            "classes": classes,
            "input_size": list(cfg["preprocess"]["resize"]) if not isinstance(cfg["preprocess"]["resize"], int) else cfg["preprocess"]["resize"],
            "normalize": tc.get("normalize", "imagenet"),
            "preprocess_version": cfg["preprocess"]["version"],
            "preprocess_fingerprint": preprocess_fingerprint(cfg),
            "run_id": run_id,
            "val_metrics": best_metrics,
            "threshold": 0.5,  # provisional; Phase 4 tunes it on val and bakes it in
        },
        model_dir / "model.pt",
    )

    run_dir = resolve_path(cfg, "artifacts_dir") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    summary = {
        "run_id": run_id,
        "smoke": smoke,
        "best_epoch_stage2": stopper.best_epoch,
        "val_metrics": best_metrics,
        "class_weights": [float(w) for w in weights],
        "wall_time_min": round(wall_min, 1),
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\ndone in {wall_min:.1f} min. best val: {best_metrics}")
    print(f"model -> {model_dir / 'model.pt'}")
    print(f"artifacts -> {run_dir}")
    return run_id


def main() -> None:
    cfg = load_config()
    train_run(cfg, smoke="--smoke" in sys.argv[1:])


if __name__ == "__main__":
    main()
