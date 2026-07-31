"""
Kinematic LSTM — Rash-Behaviour Early-Warning Classifier
=========================================================
Trains a bidirectional GRU on the annotation JSON files produced by
collect_dataset.py.  No cameras or LiDAR needed — kinematics only.

Usage
-----
    python train_kinematic_lstm.py

Outputs
-------
    models/kinematic_gru_best.pt   -- best checkpoint (by val F1)
    models/training_log.json       -- per-epoch metrics
    models/confusion_matrix.png    -- test-set confusion matrix
    models/roc_curve.png           -- test-set ROC / AUROC

Requirements
------------
    pip install torch scikit-learn matplotlib seaborn
"""

import json, os, glob, math, random, sys
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_ROOT    = "metadrive_fusion_dataset"
MODEL_DIR       = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Sequence window
SEQ_LEN         = 10      # frames of history (= 1 s at 10 FPS)
TOP_K_OBJECTS   = 3       # nearest NPCs to include per frame

# Features per frame:
#   ego:     speed, sin(heading), cos(heading), n_objects      → 4
#   per NPC: dx, dy, speed, sin(heading_ego), cos(heading_ego) → 5 × TOP_K
FEAT_DIM        = 4 + 5 * TOP_K_OBJECTS    # 4 + 15 = 19

# Training
BATCH_SIZE      = 64
EPOCHS          = 80
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
DROPOUT         = 0.35
HIDDEN_DIM      = 64
N_LAYERS        = 2

# Episode-level train / test split
# Adjust these if you collect more episodes.
# Hold out ~25% of episodes (episode-level to avoid temporal leakage).
TEST_EPISODES   = {9, 10}    # episodes with events in the test set
VAL_EPISODES    = {3, 6}     # validation during training

SEED            = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_frame_features(ann: dict) -> np.ndarray:
    """
    Convert one annotation dict → float32 feature vector of length FEAT_DIM.

    Ego features (4):
        speed_ms, sin(heading_rad), cos(heading_rad), n_nearby_objects

    Per-NPC features (5 × TOP_K_OBJECTS), sorted by distance to ego:
        dx, dy (ego-centric metres)
        speed_ms
        sin(heading_rad_ego), cos(heading_rad_ego)

    Missing NPCs (fewer than TOP_K) are zero-padded.
    """
    feat = np.zeros(FEAT_DIM, dtype=np.float32)

    ego   = ann["ego"]
    spd   = float(ego.get("speed_ms", 0.0))
    head  = float(ego.get("heading_rad", 0.0))
    objs  = ann.get("objects", [])
    n_obj = min(len(objs), 20)   # clamp for the n_objects feature

    # Normalise ego speed (typical urban max ~20 m/s)
    feat[0] = spd / 20.0
    feat[1] = math.sin(head)
    feat[2] = math.cos(head)
    feat[3] = n_obj / 10.0   # normalise object count

    # Sort objects by distance and take TOP_K
    sorted_objs = sorted(objs, key=lambda o: o.get("distance_to_ego", 999))
    for i, obj in enumerate(sorted_objs[:TOP_K_OBJECTS]):
        base = 4 + i * 5
        pos_ego  = obj.get("position_ego", [0.0, 0.0, 0.0])
        dx, dy   = float(pos_ego[0]), float(pos_ego[1])
        obj_spd  = float(obj.get("speed_ms", 0.0))
        obj_head = float(obj.get("heading_rad_ego", 0.0))

        feat[base + 0] = np.clip(dx / 60.0, -1.0, 1.0)      # ±60 m range
        feat[base + 1] = np.clip(dy / 60.0, -1.0, 1.0)
        feat[base + 2] = obj_spd / 20.0
        feat[base + 3] = math.sin(obj_head)
        feat[base + 4] = math.cos(obj_head)

    return feat


def load_clip(clip_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one event clip.

    Returns
    -------
    features : (T, FEAT_DIM) float32
    labels   : (T,) int  — 1=event frame, 0=normal
    """
    ann_files = sorted(glob.glob(os.path.join(clip_dir, "ann", "*.json")))
    if not ann_files:
        return None, None

    features = []
    labels   = []
    for af in ann_files:
        with open(af) as f:
            d = json.load(f)
        features.append(extract_frame_features(d))
        labels.append(int(d["anomaly"]["is_event_frame"]))

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_dataset(dataset_root: str) -> List[dict]:
    """
    Walk the dataset root and return a list of clip dicts:
    {
        "episode":  int,
        "clip":     str,
        "features": (T, FEAT_DIM) float32,
        "labels":   (T,) int64,
    }
    """
    clips = []
    ep_dirs = sorted(glob.glob(os.path.join(dataset_root, "episode_*")))
    total_frames = event_frames = 0

    for ep_dir in ep_dirs:
        ep_idx = int(os.path.basename(ep_dir).split("_")[1])
        for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "event_*"))):
            feats, labels = load_clip(ev_dir)
            if feats is None or len(feats) < SEQ_LEN + 1:
                continue
            clips.append({
                "episode":  ep_idx,
                "clip":     os.path.basename(ev_dir),
                "features": feats,
                "labels":   labels,
            })
            total_frames  += len(labels)
            event_frames  += labels.sum()

    n_normal = total_frames - event_frames
    print(f"Loaded {len(clips)} clips | {total_frames} frames "
          f"({event_frames} event / {n_normal} normal = "
          f"{100*event_frames/max(total_frames,1):.1f}% event)")
    return clips


# ─────────────────────────────────────────────
# DATASET / DATALOADER
# ─────────────────────────────────────────────

class KinematicSequenceDataset(Dataset):
    """
    Sliding-window dataset.  For each valid frame index i in a clip,
    yield (window[i-SEQ_LEN:i], label[i]).
    The window is zero-padded at the start of each clip.
    """

    def __init__(self, clips: List[dict]):
        self.samples = []   # (features_window, label)

        for clip in clips:
            feats  = clip["features"]   # (T, FEAT_DIM)
            labels = clip["labels"]     # (T,)
            T      = len(labels)

            for t in range(T):
                start = t - SEQ_LEN + 1
                if start < 0:
                    pad_len  = -start
                    pad      = np.zeros((pad_len, FEAT_DIM), dtype=np.float32)
                    window   = np.concatenate([pad, feats[:t+1]], axis=0)
                else:
                    window   = feats[start:t+1]

                self.samples.append((
                    torch.from_numpy(window),       # (SEQ_LEN, FEAT_DIM)
                    torch.tensor(labels[t], dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_loaders(clips: List[dict]) -> tuple:
    """
    Split clips by episode index → train / val / test DataLoaders.
    Returns (train_loader, val_loader, test_loader, pos_weight_tensor).
    """
    train_clips = [c for c in clips if c["episode"] not in TEST_EPISODES | VAL_EPISODES]
    val_clips   = [c for c in clips if c["episode"] in VAL_EPISODES]
    test_clips  = [c for c in clips if c["episode"] in TEST_EPISODES]

    print(f"\nSplit — train clips: {len(train_clips)} | "
          f"val clips: {len(val_clips)} | "
          f"test clips: {len(test_clips)}")

    # Compute class weight for weighted BCE
    all_labels = np.concatenate([c["labels"] for c in train_clips])
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    # pos_weight = n_neg / n_pos  (down-weights the majority event class)
    pos_weight = float(n_neg) / max(float(n_pos), 1.0)
    print(f"pos_weight (BCE) = {pos_weight:.3f}  "
          f"(train: {int(n_pos)} event / {int(n_neg)} normal frames)")

    train_ds = KinematicSequenceDataset(train_clips)
    val_ds   = KinematicSequenceDataset(val_clips)
    test_ds  = KinematicSequenceDataset(test_clips)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pw = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    return train_loader, val_loader, test_loader, pw


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class RashBehaviourGRU(nn.Module):
    """
    Bidirectional GRU for frame-level rash-behaviour classification.

    Input  : (batch, seq_len, feat_dim)
    Output : (batch,) — raw logit (apply sigmoid for probability)
    """

    def __init__(
        self,
        input_dim:  int = FEAT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_layers:   int = N_LAYERS,
        dropout:    float = DROPOUT,
    ):
        super().__init__()

        # Input projection + normalisation
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Bidirectional GRU
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feat_dim)
        x   = self.input_proj(x)          # (B, T, H)
        out, _ = self.gru(x)              # (B, T, 2H)
        last = out[:, -1, :]              # (B, 2H) — last timestep
        return self.head(last).squeeze(-1)  # (B,)


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimiser=None):
    """One train or eval epoch.  Returns (loss, all_probs, all_labels)."""
    is_train = optimiser is not None
    model.train(is_train)
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = criterion(logits, y)

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            total_loss += loss.item() * len(y)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / max(len(loader.dataset), 1)
    return avg_loss, all_probs, all_labels


def threshold_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    f1    = f1_score(labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {"f1": f1, "precision": prec, "recall": rec, "auc": auc}


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_confusion(labels, probs, threshold=0.5, save_path="models/confusion_matrix.png"):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Rash"],
                yticklabels=["Normal", "Rash"], ax=ax)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc(labels, probs, auc_val, save_path="models/roc_curve.png"):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(log, save_path="models/training_curves.png"):
    epochs  = [e["epoch"] for e in log]
    train_f = [e["train_f1"] for e in log]
    val_f   = [e["val_f1"] for e in log]
    train_l = [e["train_loss"] for e in log]
    val_l   = [e["val_loss"] for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_l, label="train"); ax1.plot(epochs, val_l, label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (weighted BCE)"); ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(epochs, train_f, label="train"); ax2.plot(epochs, val_f, label="val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1"); ax2.set_title("F1 Score")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# FIND OPTIMAL THRESHOLD
# ─────────────────────────────────────────────

def find_best_threshold(probs, labels):
    """Sweep thresholds on the validation set and return the one maximising F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.05):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load data ──
    clips = load_dataset(DATASET_ROOT)
    if not clips:
        print("ERROR: No clips found. Check DATASET_ROOT path.")
        sys.exit(1)

    train_loader, val_loader, test_loader, pos_weight = make_loaders(clips)

    if len(train_loader.dataset) == 0:
        print("ERROR: Train set is empty.  Adjust TEST_EPISODES / VAL_EPISODES.")
        sys.exit(1)

    # ── Model + optimiser ──
    model     = RashBehaviourGRU().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: RashBehaviourGRU  |  Parameters: {n_params:,}")
    print(f"Training for {EPOCHS} epochs on {DEVICE}\n")

    best_val_f1   = -1.0
    best_ckpt     = os.path.join(MODEL_DIR, "kinematic_gru_best.pt")
    training_log  = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_probs, tr_labels = run_epoch(model, train_loader, criterion, optimiser)
        vl_loss, vl_probs, vl_labels = run_epoch(model, val_loader,   criterion)
        scheduler.step()

        tr_m = threshold_metrics(tr_probs, tr_labels)
        vl_m = threshold_metrics(vl_probs, vl_labels)

        log_entry = {
            "epoch":      epoch,
            "train_loss": round(tr_loss, 5),
            "train_f1":   round(tr_m["f1"], 4),
            "val_loss":   round(vl_loss, 5),
            "val_f1":     round(vl_m["f1"], 4),
            "val_auc":    round(vl_m["auc"], 4),
        }
        training_log.append(log_entry)

        if vl_m["f1"] > best_val_f1:
            best_val_f1 = vl_m["f1"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      best_val_f1,
                "feat_dim":    FEAT_DIM,
                "seq_len":     SEQ_LEN,
                "top_k":       TOP_K_OBJECTS,
            }, best_ckpt)
            star = " << best"
        else:
            star = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train: loss={tr_loss:.4f} F1={tr_m['f1']:.3f}  |  "
                  f"val:   loss={vl_loss:.4f} F1={vl_m['f1']:.3f} "
                  f"AUC={vl_m['auc']:.3f}{star}")

    print(f"\nBest val F1: {best_val_f1:.4f}  (checkpoint: {best_ckpt})")

    # ── Save training log ──
    log_path = os.path.join(MODEL_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    plot_training_curves(training_log)

    # ── Test evaluation ──
    print("\n--- Test set evaluation ---")
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    _, test_probs, test_labels = run_epoch(model, test_loader, criterion)

    if test_labels.sum() == 0:
        print("WARNING: No event frames in the test episodes — "
              "adjust TEST_EPISODES to include episodes with events.")
    else:
        # Find optimal threshold on val set, apply to test
        _, val_probs, val_labels = run_epoch(model, val_loader, criterion)
        best_t, _ = find_best_threshold(val_probs, val_labels)
        print(f"Optimal threshold (from val set): {best_t:.2f}")

        test_m = threshold_metrics(test_probs, test_labels, threshold=best_t)
        print(f"\n  F1        : {test_m['f1']:.4f}")
        print(f"  Precision : {test_m['precision']:.4f}")
        print(f"  Recall    : {test_m['recall']:.4f}")
        print(f"  AUROC     : {test_m['auc']:.4f}")

        print("\nClassification report:")
        preds = (test_probs >= best_t).astype(int)
        print(classification_report(test_labels, preds,
                                    target_names=["Normal", "Rash Event"],
                                    zero_division=0))

        plot_confusion(test_labels, test_probs, threshold=best_t)
        try:
            plot_roc(test_labels, test_probs, test_m["auc"])
        except Exception as e:
            print(f"ROC plot skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
