"""
TTC (Time-To-Collision) evaluation for the Early Fusion v2 model.

For every test-set window, computes the minimum kinematic TTC across all
nearby vehicles and aligns it with the model prediction. Reports:

  1. Mean TTC at detection (TP frames) vs missed detections (FN frames)
  2. TTC distribution by prediction class via violin plot (TP / TN / FP / FN)
  3. TTC-Threshold curve: mean TTC at detection across thresholds 0.05->0.95
  4. Dangerous miss rate: FN frames where TTC < DANGER_TTC seconds

TTC formula (world frame, per NPC):
    closing_speed = -dot(v_npc - v_ego, unit_vector_ego_to_npc)
    TTC = distance / closing_speed   (only when closing_speed > CLOSE_MIN)

Usage:
    cd simulation
    python eval_ttc.py
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_early_fusion_v2 import (
    EarlyFusionV2, EarlyFusionDatasetV2, FusionClip,
    MODEL_DIR, DEVICE, CACHE_ROOT, DATASET_ROOT,
    SEQ_LEN, TEST_EPISODES, VAL_EPISODES,
    find_best_threshold,
)
from train_kinematic_lstm import extract_frame_features as extract_json_features

BATCH_SIZE  = 32
TTC_CAP     = 30.0   # seconds -- cap for non-converging pairs; used as "inf" placeholder
DANGER_TTC  = 3.0    # seconds -- threshold below which a miss is "safety-critical"
CLOSE_MIN   = 0.5    # m/s    -- minimum closing speed to count as approaching


# =============================================================================
# TTC computation
# =============================================================================

def compute_min_ttc(ann: dict) -> float:
    """
    Minimum kinematic TTC (seconds) over all nearby vehicles in one frame.

    Uses world-frame velocity vectors (not speed scalars) so that closing speed
    is correctly resolved along the ego-to-NPC direction.

    Returns TTC_CAP when no vehicle is on a converging trajectory.
    """
    try:
        ego_pos = np.array(ann["ego"]["position_world"][:2], dtype=np.float64)
        ego_vel = np.array(ann["ego"]["velocity"][:2],       dtype=np.float64)
    except (KeyError, TypeError):
        return TTC_CAP

    min_ttc = TTC_CAP
    for obj in ann.get("objects", []):
        dist = float(obj.get("distance_to_ego", 0.0))
        if dist < 0.5:          # skip overlapping / same-position objects
            continue
        try:
            npc_pos = np.array(obj["position_world"][:2], dtype=np.float64)
            npc_vel = np.array(obj["velocity"][:2],       dtype=np.float64)
        except (KeyError, TypeError):
            continue

        d_vec   = npc_pos - ego_pos                        # world-frame direction
        d_hat   = d_vec / (np.linalg.norm(d_vec) + 1e-9)  # unit vector ego -> NPC
        v_rel   = npc_vel - ego_vel                        # NPC velocity rel. to ego
        closing = -np.dot(v_rel, d_hat)                   # +ve means converging

        if closing > CLOSE_MIN:
            ttc = min(dist / closing, TTC_CAP)
            min_ttc = min(min_ttc, ttc)

    return min_ttc


# =============================================================================
# Data loading -- clips + per-frame TTC arrays
# =============================================================================

def load_subset_with_ttc(episode_set: set):
    """
    Same as eval_ef2.load_subset but also computes a TTC value per frame.
    Returns (clips, ttc_arrays) where ttc_arrays[i] is float32 of length len(clips[i].labels).
    """
    clips      = []
    ttc_arrays = []

    for ep_dir in sorted(glob.glob(f"{DATASET_ROOT}/episode_*")):
        ep_idx = int(os.path.basename(ep_dir).split("_")[1])
        if ep_idx not in episode_set:
            continue

        for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "event_*"))):
            ev_name   = os.path.basename(ev_dir)
            cache_dir = os.path.join(CACHE_ROOT, os.path.basename(ep_dir), ev_name)
            ann_files = sorted(glob.glob(os.path.join(ev_dir, "ann", "*.json")))
            if not ann_files:
                continue

            rgb_l=[]; sem_l=[]; bev_l=[]; json_l=[]; lbl_l=[]; ttc_l=[]
            ok = True
            for af in ann_files:
                fid   = os.path.splitext(os.path.basename(af))[0]
                cpath = os.path.join(cache_dir, f"{fid}.npz")
                if not os.path.exists(cpath):
                    ok = False; break
                c = np.load(cpath)
                rgb_l.append(c["rgb_feat"].astype(np.float32))
                sem_l.append(c["sem_feat"].astype(np.float32))
                bev_l.append(c["bev"])   # keep float16 to match training
                with open(af) as f:
                    ann = json.load(f)
                json_l.append(extract_json_features(ann))
                lbl_l.append(int(ann["anomaly"]["is_event_frame"]))
                ttc_l.append(compute_min_ttc(ann))

            if not ok or len(lbl_l) < SEQ_LEN + 1:
                continue

            clips.append(FusionClip(
                episode    = ep_idx,
                rgb_feats  = np.array(rgb_l,  dtype=np.float32),
                sem_feats  = np.array(sem_l,  dtype=np.float32),
                bev_grids  = np.array(bev_l,  dtype=np.float16),
                json_feats = np.array(json_l, dtype=np.float32),
                labels     = np.array(lbl_l,  dtype=np.int64),
            ))
            ttc_arrays.append(np.array(ttc_l, dtype=np.float32))

    return clips, ttc_arrays


def extract_window_ttcs(clips, ttc_arrays) -> np.ndarray:
    """
    For each sample produced by EarlyFusionDatasetV2 (with shuffle=False),
    return the TTC of frame t (the classified frame).

    EarlyFusionDatasetV2 iterates: for t in range(T) with zero-padding for
    early frames, so the per-sample TTC is simply ttc_arr[t] for t in range(T).
    """
    out = []
    for clip, ttc_arr in zip(clips, ttc_arrays):
        for t in range(len(clip.labels)):
            out.append(float(ttc_arr[t]))
    return np.array(out, dtype=np.float32)


# =============================================================================
# Inference
# =============================================================================

def infer(model, clips):
    ds = EarlyFusionDatasetV2(clips)
    dl = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for x_rgb, x_sem, x_bev, x_json, y in dl:
            logits = model(x_rgb.to(DEVICE), x_sem.to(DEVICE),
                           x_bev.to(DEVICE), x_json.to(DEVICE))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# =============================================================================
# Plotting helpers
# =============================================================================

DARK_BG = "#0d1117"
C_TP    = "#3fb950"   # green  -- correct event detection
C_TN    = "#58a6ff"   # blue   -- correct normal
C_FP    = "#e3b341"   # amber  -- false alarm
C_FN    = "#f78166"   # red    -- missed event (dangerous)
GREY    = "#8b949e"


def _apply_dark_style(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=GREY)
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")


def plot_ttc_distribution(ttc_disp, preds, labels, save_path):
    """Violin plot of TTC values split by prediction outcome."""
    groups = {
        "TP": ttc_disp[(preds == 1) & (labels == 1)],
        "FP": ttc_disp[(preds == 1) & (labels == 0)],
        "FN": ttc_disp[(preds == 0) & (labels == 1)],
        "TN": ttc_disp[(preds == 0) & (labels == 0)],
    }
    colors = {"TP": C_TP, "FP": C_FP, "FN": C_FN, "TN": C_TN}

    # Drop empty groups
    valid = {k: v for k, v in groups.items() if len(v) > 0}

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(
        list(valid.values()),
        positions=range(len(valid)),
        showmedians=True,
        showextrema=True,
    )
    for pc, key in zip(parts["bodies"], valid):
        pc.set_facecolor(colors[key])
        pc.set_alpha(0.7)
    for comp in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[comp].set_edgecolor(GREY)

    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([f"{k}\nn={len(v)}" for k, v in valid.items()])
    ax.set_ylabel("Min TTC (s)")
    ax.set_title("TTC Distribution by Prediction Type -- EF-v2 Test Set")
    ax.axhline(DANGER_TTC, color=C_FN, ls="--", lw=1.4,
               label=f"Danger threshold ({DANGER_TTC} s)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=GREY)
    _apply_dark_style(fig, ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_threshold_curve(ttc_disp, probs, labels, save_path):
    """Mean TTC at detection vs classification threshold, with TP count bar overlay."""
    thresholds = np.arange(0.05, 0.96, 0.05)
    mean_ttcs, n_tps = [], []

    for t in thresholds:
        tp_mask = (probs >= t) & (labels == 1)
        if tp_mask.sum() > 0:
            mean_ttcs.append(float(ttc_disp[tp_mask].mean()))
            n_tps.append(int(tp_mask.sum()))
        else:
            mean_ttcs.append(np.nan)
            n_tps.append(0)

    mean_ttcs = np.array(mean_ttcs, dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, mean_ttcs, color=C_TP, lw=2.2, marker="o", ms=5,
             label="Mean TTC at detection (s)")
    ax1.axhline(DANGER_TTC, color=C_FN, ls="--", lw=1.4,
                label=f"Danger threshold ({DANGER_TTC} s)")
    ax2.bar(thresholds, n_tps, width=0.04, alpha=0.25,
            color=C_TN, label="TP count")

    ax1.set_xlabel("Classification Threshold")
    ax1.set_ylabel("Mean TTC at Detection (s)", color=C_TP)
    ax2.set_ylabel("True Positive Count", color=C_TN)
    ax1.set_title("TTC-Threshold Curve -- Safety Margin vs Recall Trade-off")
    ax1.set_ylim(0, TTC_CAP + 2)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               facecolor="#161b22", edgecolor="#30363d", labelcolor=GREY)
    _apply_dark_style(fig, [ax1, ax2])
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_dangerous_misses(ttc_disp, preds, labels, save_path):
    """TTC histogram comparing correct detections (TP) vs missed events (FN)."""
    tp_ttcs = ttc_disp[(preds == 1) & (labels == 1)]
    fn_ttcs = ttc_disp[(preds == 0) & (labels == 1)]

    bins = np.linspace(0, TTC_CAP, 31)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tp_ttcs, bins=bins, alpha=0.65, color=C_TP,
            label=f"Detected (TP, n={len(tp_ttcs)})")
    ax.hist(fn_ttcs, bins=bins, alpha=0.65, color=C_FN,
            label=f"Missed (FN, n={len(fn_ttcs)})")
    ax.axvline(DANGER_TTC, color="white", ls="--", lw=1.5,
               label=f"Safety-critical zone (TTC < {DANGER_TTC} s)")
    ax.set_xlabel("Min TTC (s)")
    ax.set_ylabel("Frame count")
    ax.set_title("Detected vs Missed Events by TTC -- Safety-Critical Analysis")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=GREY)
    _apply_dark_style(fig, ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# =============================================================================
# Main
# =============================================================================

def main():
    ckpt_path = os.path.join(MODEL_DIR, "ef2_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}. Train first.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"Checkpoint  : epoch={ckpt['epoch']}, val_F1={ckpt['val_f1']:.4f}")

    model = EarlyFusionV2().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # ── Validation set: find optimal threshold ────────────────────────────────
    print("\nLoading val clips for threshold search...")
    val_clips, _ = load_subset_with_ttc(VAL_EPISODES)
    print(f"  {len(val_clips)} val clips loaded")
    vl_p, vl_l   = infer(model, val_clips)
    best_t, best_f1 = find_best_threshold(vl_p, vl_l)
    print(f"  Optimal threshold : {best_t:.2f}   val F1 = {best_f1:.4f}")

    # ── Test set ──────────────────────────────────────────────────────────────
    print("\nLoading test clips + computing per-frame TTC...")
    test_clips, test_ttcs = load_subset_with_ttc(TEST_EPISODES)
    print(f"  {len(test_clips)} test clips loaded")
    ts_p, ts_l = infer(model, test_clips)

    # Align TTC values with DataLoader output order (shuffle=False is critical)
    ttc_raw  = extract_window_ttcs(test_clips, test_ttcs)   # shape: [N]
    ts_preds = (ts_p >= best_t).astype(int)

    if len(ttc_raw) != len(ts_p):
        print(f"ERROR: TTC length {len(ttc_raw)} != inference length {len(ts_p)}")
        return

    # Substitute TTC_CAP for non-approaching frames (TTC = inf)
    ttc_disp = np.where(ttc_raw < TTC_CAP, ttc_raw, TTC_CAP)

    # ── Statistics ────────────────────────────────────────────────────────────
    n_total       = len(ts_p)
    n_approaching = (ttc_raw < TTC_CAP).sum()
    tp_mask       = (ts_preds == 1) & (ts_l == 1)
    fn_mask       = (ts_preds == 0) & (ts_l == 1)
    fp_mask       = (ts_preds == 1) & (ts_l == 0)
    tn_mask       = (ts_preds == 0) & (ts_l == 0)
    total_events  = (ts_l == 1).sum()

    print(f"\n{'='*55}")
    print(f"  TTC Evaluation -- EarlyFusionV2 -- Test Set")
    print(f"{'='*55}")
    print(f"  Total test windows         : {n_total}")
    print(f"  Windows with approaching   : {n_approaching} "
          f"({100*n_approaching/n_total:.1f}%)")
    print()

    def _ttc_stats(mask, name):
        arr = ttc_disp[mask]
        if arr.size == 0:
            print(f"  {name:20s}: n=0")
            return
        print(f"  {name:20s}: n={arr.size:4d} | "
              f"mean={arr.mean():5.2f}s | "
              f"median={np.median(arr):5.2f}s | "
              f"min={arr.min():5.2f}s")

    _ttc_stats(tp_mask, "True Positives")
    _ttc_stats(fp_mask, "False Positives")
    _ttc_stats(fn_mask, "False Negatives")
    _ttc_stats(tn_mask, "True Negatives")

    # Dangerous misses: missed events where a vehicle was already very close
    dangerous_fn = fn_mask & (ttc_disp < DANGER_TTC)
    print(f"\n  Dangerous misses (FN + TTC<{DANGER_TTC}s): "
          f"{dangerous_fn.sum()} / {total_events} event frames "
          f"({100*dangerous_fn.sum()/max(total_events,1):.1f}%)")

    if tp_mask.sum() > 0:
        margin = ttc_disp[tp_mask].mean()
        print(f"\n  >> Model raises alert with avg {margin:.2f} s TTC remaining")
        print(f"     (threshold={best_t:.2f}, optimised on val-F1)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating TTC plots...")
    plot_ttc_distribution(
        ttc_disp, ts_preds, ts_l,
        os.path.join(MODEL_DIR, "nb_ttc_distribution.png"),
    )
    plot_threshold_curve(
        ttc_disp, ts_p, ts_l,
        os.path.join(MODEL_DIR, "nb_ttc_threshold_curve.png"),
    )
    plot_dangerous_misses(
        ttc_disp, ts_preds, ts_l,
        os.path.join(MODEL_DIR, "nb_ttc_dangerous_misses.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
