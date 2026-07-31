"""
Kinematic LSTM — Frame-by-frame inference demo
================================================
Loads a trained checkpoint and runs it over every annotated frame in a
given episode, printing a timeline of predicted rash-event probability.

Usage
-----
    python infer_kinematic_lstm.py --episode 9
    python infer_kinematic_lstm.py --episode 9 --threshold 0.4

Outputs a text timeline and saves a probability-over-time plot.
"""

import argparse, json, glob, os, math
import numpy as np
import torch
import torch.nn as nn

# ── import helpers from training script ────────────────────────────────────────
import sys; sys.path.insert(0, os.path.dirname(__file__))
from train_kinematic_lstm import (
    RashBehaviourGRU, extract_frame_features,
    FEAT_DIM, SEQ_LEN, MODEL_DIR, DATASET_ROOT,
)

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: str) -> tuple:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = RashBehaviourGRU(
        input_dim  = ckpt.get("feat_dim", FEAT_DIM),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def infer_episode(episode_idx: int, threshold: float = 0.5, ckpt_path: str = None):
    if ckpt_path is None:
        ckpt_path = os.path.join(MODEL_DIR, "kinematic_gru_best.pt")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("Run  python train_kinematic_lstm.py  first.")
        return

    model, ckpt = load_model(ckpt_path)
    print(f"Loaded checkpoint (epoch={ckpt['epoch']}, val_F1={ckpt['val_f1']:.4f})")

    ep_dir = os.path.join(DATASET_ROOT, f"episode_{episode_idx:04d}")
    ann_files = sorted(glob.glob(os.path.join(ep_dir, "event_*/ann/*.json")))

    if not ann_files:
        print(f"No annotation files found in {ep_dir}")
        return

    print(f"\nEpisode {episode_idx:04d} — {len(ann_files)} frames\n")

    # Build feature matrix + labels
    feats  = []
    labels = []
    steps  = []
    for af in ann_files:
        with open(af) as f:
            d = json.load(f)
        feats.append(extract_frame_features(d))
        labels.append(int(d["anomaly"]["is_event_frame"]))
        steps.append(d["step"])

    feats  = np.array(feats,  dtype=np.float32)   # (T, FEAT_DIM)
    labels = np.array(labels, dtype=np.int64)
    T      = len(labels)

    # Sliding-window inference
    probs  = np.zeros(T, dtype=np.float32)
    seq_len = ckpt.get("seq_len", SEQ_LEN)

    with torch.no_grad():
        for t in range(T):
            start = t - seq_len + 1
            if start < 0:
                pad    = np.zeros((-start, FEAT_DIM), dtype=np.float32)
                window = np.concatenate([pad, feats[:t+1]], axis=0)
            else:
                window = feats[start:t+1]
            x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)   # (1, T, F)
            logit    = model(x)
            probs[t] = torch.sigmoid(logit).item()

    preds = (probs >= threshold).astype(int)

    # ── Print timeline ──
    header = f"{'Frame':>6}  {'Step':>5}  {'True':>6}  {'Prob':>6}  {'Pred':>6}  {'Match':>6}"
    print(header)
    print("─" * len(header))
    for t in range(T):
        match = "✓" if preds[t] == labels[t] else "✗"
        flag  = " ← EVENT" if labels[t] == 1 and preds[t] == 1 else (
                " ← MISS"  if labels[t] == 1 and preds[t] == 0 else (
                " ← FP"    if labels[t] == 0 and preds[t] == 1 else ""))
        print(f"{t:6d}  {steps[t]:5d}  {labels[t]:6d}  {probs[t]:6.3f}  {preds[t]:6d}  {match:>6}{flag}")

    # ── Summary ──
    from sklearn.metrics import f1_score, roc_auc_score
    f1  = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    print(f"\nF1={f1:.4f}  AUROC={auc:.4f}  threshold={threshold:.2f}")

    # ── Plot ──
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(12, 4))
        t_axis = list(range(T))
        ax.plot(t_axis, probs, lw=1.5, label="P(rash)", color="steelblue")
        ax.fill_between(t_axis,
                        [l * 0.9 for l in labels], 0,
                        alpha=0.25, color="red", label="True event")
        ax.axhline(threshold, ls="--", lw=1, color="orange", label=f"Threshold {threshold:.2f}")
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frame index"); ax.set_ylabel("P(rash event)")
        ax.set_title(f"Episode {episode_idx:04d} — Kinematic LSTM prediction")
        ax.legend()
        plt.tight_layout()
        out = os.path.join(MODEL_DIR, f"infer_ep{episode_idx:04d}.png")
        plt.savefig(out, dpi=120)
        print(f"Plot saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode",   type=int,   default=9)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ckpt",      type=str,   default=None)
    args = p.parse_args()
    infer_episode(args.episode, threshold=args.threshold, ckpt_path=args.ckpt)
