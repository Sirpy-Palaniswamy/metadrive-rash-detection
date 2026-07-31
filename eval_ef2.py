"""
Memory-efficient test-set evaluation for the Early Fusion v2 best checkpoint.
Loads ONLY the val and test clips (not training clips) to stay within RAM limits.

Usage:  python eval_ef2.py
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import everything from train_ef2 (needed for model + dataset classes)
from train_early_fusion_v2 import (
    EarlyFusionV2, EarlyFusionDatasetV2, FusionClip,
    MODEL_DIR, DEVICE, CACHE_ROOT, DATASET_ROOT,
    SEQ_LEN, CAM_BB_DIM, JSON_DIM,
    TEST_EPISODES, VAL_EPISODES,
    find_best_threshold, metrics,
)
from train_kinematic_lstm import extract_frame_features as extract_json_features

BATCH_SIZE = 32

# ─── Load only val + test clips ──────────────────────────────────────────────

def load_subset(episode_set) -> list:
    """Load only the clips from the given episode set (BEV as float16 in RAM)."""
    clips_out = []
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
            rgb_l=[]; sem_l=[]; bev_l=[]; json_l=[]; lbl_l=[]
            ok = True
            for af in ann_files:
                fid   = os.path.splitext(os.path.basename(af))[0]
                cpath = os.path.join(cache_dir, f"{fid}.npz")
                if not os.path.exists(cpath):
                    ok = False; break
                c = np.load(cpath)
                rgb_l.append(c["rgb_feat"].astype(np.float32))
                sem_l.append(c["sem_feat"].astype(np.float32))
                bev_l.append(c["bev"])   # keep float16
                with open(af) as f:
                    ann = json.load(f)
                json_l.append(extract_json_features(ann))
                lbl_l.append(int(ann["anomaly"]["is_event_frame"]))
            if not ok or len(lbl_l) < SEQ_LEN + 1:
                continue
            clips_out.append(FusionClip(
                episode    = ep_idx,
                rgb_feats  = np.array(rgb_l,  dtype=np.float32),
                sem_feats  = np.array(sem_l,  dtype=np.float32),
                bev_grids  = np.array(bev_l,  dtype=np.float16),
                json_feats = np.array(json_l, dtype=np.float32),
                labels     = np.array(lbl_l,  dtype=np.int64),
            ))
    return clips_out


# ─── Inference pass ──────────────────────────────────────────────────────────

def infer(model, clips):
    ds = EarlyFusionDatasetV2(clips)
    dl = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    all_probs  = []; all_labels = []
    model.eval()
    with torch.no_grad():
        for x_rgb, x_sem, x_bev, x_json, y in dl:
            logits = model(x_rgb.to(DEVICE), x_sem.to(DEVICE),
                          x_bev.to(DEVICE), x_json.to(DEVICE))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ckpt_path = os.path.join(MODEL_DIR, "ef2_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path}. Train first."); return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']}, val_F1={ckpt['val_f1']:.4f}")

    model = EarlyFusionV2().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Load val + test separately
    print("Loading val clips...")
    val_clips  = load_subset(VAL_EPISODES)
    print(f"  {len(val_clips)} val clips loaded")

    print("Loading test clips...")
    test_clips = load_subset(TEST_EPISODES)
    print(f"  {len(test_clips)} test clips loaded")

    vl_p, vl_l = infer(model, val_clips)
    ts_p, ts_l = infer(model, test_clips)

    best_t, best_f1_val = find_best_threshold(vl_p, vl_l)
    print(f"\nOptimal threshold (val): {best_t:.2f}  val F1={best_f1_val:.4f}")

    if ts_l.sum() == 0:
        print("WARNING: No event frames in test clips.")
        return

    ts_m = metrics(ts_p, ts_l, t=best_t)
    print(f"\n--- Test set results (EarlyFusionV2, epoch {ckpt['epoch']}) ---")
    print(f"  F1        : {ts_m['f1']:.4f}")
    print(f"  Precision : {ts_m['precision']:.4f}")
    print(f"  Recall    : {ts_m['recall']:.4f}")
    print(f"  AUROC     : {ts_m['auc']:.4f}")
    print()
    print(classification_report(
        ts_l, (ts_p >= best_t).astype(int),
        target_names=["Normal","Rash Event"], zero_division=0))

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(ts_l, ts_p)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(fpr, tpr, lw=2,
                label=f"EarlyFusion v2  AUROC={ts_m['auc']:.3f}")
        ax.plot([0,1],[0,1],"k--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve -- Early Fusion v2 (Test)")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "ef2_roc_curve.png"), dpi=120)
        plt.close(); print("  Saved: models/ef2_roc_curve.png")
    except Exception as e:
        print(f"  ROC skipped: {e}")

    # Comparison table
    print("\n--- Model comparison (val F1 from saved checkpoint) ---")
    for fname, label in [
        ("kinematic_gru_best.pt", "Kinematics only               "),
        ("early_fusion_best.pt",  "Early Fusion v1 (RGB+Kin+LiDAR)"),
        ("ef2_best.pt",           "Early Fusion v2 (BEV+RGB+Sem+J)"),
    ]:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            c = torch.load(p, map_location="cpu", weights_only=False)
            print(f"  {label}: val F1 = {c['val_f1']:.4f}  (epoch {c['epoch']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
