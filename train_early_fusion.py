"""
Early-Fusion Multi-Modal Rash-Behaviour Classifier
====================================================
Fuses three modalities at the FRAME level before temporal reasoning:

  Camera  (RGB)   -- MobileNetV3-Small over 6 surround views (mean-pooled)
                     Frozen ImageNet backbone + trainable projection head
  Kinematics      -- 19-dim ego + NPC state vector (from annotation JSON)
  LiDAR           -- 9-dim intensity-based statistics
                     (intensity is correct; XYZ coordinates are degraded
                      in the current batch -- re-enable 3-D features after
                      running collect_dataset.py with the new 360-deg code)

Fusion strategy -- EARLY FUSION
  per-frame feature vector = concat(cam_64, kin_32, lidar_16) = 112 dims
  fed as a TIME sequence into a Bidirectional GRU

Two-phase execution
  Phase 1 (run once, ~4-5 min CPU):
      extract_and_cache_features()
      Saves  feature_cache/<ep>/<ev>/<FFFFF>.npz  (cam + lidar feats)
  Phase 2 (repeatable, ~5-10 min CPU):
      train and evaluate the BiGRU on cached + kinematic features

Usage
-----
    python train_early_fusion.py                # full run (extract + train)
    python train_early_fusion.py --skip-cache   # skip extraction, train only

Output
------
    models/early_fusion_best.pt
    models/early_fusion_training_log.json
    models/early_fusion_training_curves.png
    models/early_fusion_confusion_matrix.png
    models/early_fusion_roc_curve.png
"""

import argparse, json, os, glob, math, random, sys, warnings
from typing import List, Tuple, Optional, Dict
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Import kinematic feature extractor from existing training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_kinematic_lstm import extract_frame_features as extract_kinematics
from train_kinematic_lstm import FEAT_DIM as KIN_DIM   # 19

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_ROOT   = "metadrive_fusion_dataset"
CACHE_ROOT     = "feature_cache"
MODEL_DIR      = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

# Surround-view camera names (order matches collect_dataset.py)
CAMERA_NAMES = ["front", "front_left", "front_right",
                "back",  "back_left",  "back_right"]

# ── Feature dimensions ──
CAM_BACKBONE_DIM = 576     # MobileNetV3-Small pool output
CAM_PROJ_DIM     = 64      # trainable projection from backbone → this
KIN_PROJ_DIM     = 32      # projection of raw 19-dim kinematics
LIDAR_STAT_DIM   = 9       # intensity-based LiDAR statistics
LIDAR_PROJ_DIM   = 16      # projection of LiDAR stats
FUSED_DIM        = CAM_PROJ_DIM + KIN_PROJ_DIM + LIDAR_PROJ_DIM  # 112

# ── Sequence + model ──
SEQ_LEN    = 10
HIDDEN_DIM = 64
N_LAYERS   = 2
DROPOUT    = 0.35

# ── Training ──
BATCH_SIZE   = 32
EPOCHS       = 80
LR           = 3e-4
WEIGHT_DECAY = 1e-4

# ── Episode splits (same as kinematic model for fair comparison) ──
TEST_EPISODES = {9, 10}
VAL_EPISODES  = {3, 6}

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

# ImageNet normalisation for MobileNetV3
IMG_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# PHASE 1 — FEATURE EXTRACTION & CACHING
# ─────────────────────────────────────────────

def build_backbone() -> nn.Module:
    """
    Frozen MobileNetV3-Small up to the pooling layer.
    Returns features of shape (B, 576).
    """
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Freeze all backbone parameters
    for p in model.parameters():
        p.requires_grad = False
    # We only use .features + .avgpool
    encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
    encoder.eval()
    return encoder.to(DEVICE)


def extract_cam_feature(backbone: nn.Module, clip_dir: str,
                        frame_id: str) -> Optional[np.ndarray]:
    """
    Load all 6 surround-view RGB images for one frame, pass through
    the frozen backbone, and return the MEAN-POOLED feature vector (576,).
    Returns None if no valid images are found.
    """
    view_feats = []
    for cam in CAMERA_NAMES:
        img_path = os.path.join(clip_dir, "rgb", f"{frame_id}_{cam}.png")
        if not os.path.exists(img_path):
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = IMG_TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = backbone(tensor).squeeze(0).cpu().numpy()  # (576,)
        view_feats.append(feat)

    if not view_feats:
        return None
    return np.mean(view_feats, axis=0).astype(np.float32)   # (576,)


def extract_lidar_stats(clip_dir: str, frame_id: str) -> np.ndarray:
    """
    Compute 9 intensity-based statistics from the LiDAR point cloud.

    Why intensity only?
        The XYZ coordinates in the current batch have a world-frame offset
        (the camera was not properly constrained to ego frame when this data
        was collected). Intensity is correctly normalised to [0, 1] via the
        fixed formula 1 - dist/100 and is unaffected by the coordinate bug.
        Re-enable full 3-D statistics once LiDAR is re-collected.

    Features
    --------
    0  n_points_norm        point count / 30000
    1  intensity_mean
    2  intensity_std
    3  intensity_p10        10th percentile
    4  intensity_p25        25th percentile
    5  intensity_p75        75th percentile
    6  intensity_p90        90th percentile
    7  frac_high_intensity  fraction with intensity > 0.6  (nearby objects)
    8  z_std_norm           std(Z) / 20  (height variation; Z is less affected)
    """
    stats = np.zeros(LIDAR_STAT_DIM, dtype=np.float32)
    lidar_path = os.path.join(clip_dir, "lidar", f"{frame_id}.npy")
    if not os.path.exists(lidar_path):
        return stats

    pts = np.load(lidar_path)
    if pts.ndim != 2 or pts.shape[1] < 4 or pts.shape[0] == 0:
        return stats

    intensity = pts[:, 3].astype(np.float32)
    z_vals    = pts[:, 2].astype(np.float32)

    stats[0] = min(len(pts) / 30000.0, 1.0)
    stats[1] = float(intensity.mean())
    stats[2] = float(intensity.std())
    stats[3] = float(np.percentile(intensity, 10))
    stats[4] = float(np.percentile(intensity, 25))
    stats[5] = float(np.percentile(intensity, 75))
    stats[6] = float(np.percentile(intensity, 90))
    stats[7] = float((intensity > 0.6).mean())
    stats[8] = float(np.clip(z_vals.std() / 20.0, 0.0, 1.0))
    return stats


def extract_and_cache_features():
    """
    Phase 1: one-time pass over the entire dataset.
    Saves  CACHE_ROOT/<ep>/<ev>/<FFFFF>.npz  per frame containing:
        cam_feat   (576,)  float32 — mean-pooled MobileNetV3 backbone features
        lidar_feat (9,)    float32 — LiDAR intensity statistics
    """
    print("\n=== Phase 1: Feature Extraction & Caching ===")
    backbone = build_backbone()

    clips = sorted(glob.glob(f"{DATASET_ROOT}/episode_*/event_*"))
    total = 0

    for clip_dir in clips:
        parts   = clip_dir.replace("\\", "/").split("/")
        ep_name = parts[-2]   # e.g. episode_0000
        ev_name = parts[-1]   # e.g. event_0000

        cache_dir = os.path.join(CACHE_ROOT, ep_name, ev_name)
        os.makedirs(cache_dir, exist_ok=True)

        ann_files = sorted(glob.glob(os.path.join(clip_dir, "ann", "*.json")))
        for ann_path in ann_files:
            frame_id = os.path.splitext(os.path.basename(ann_path))[0]  # "00000"
            cache_path = os.path.join(cache_dir, f"{frame_id}.npz")

            if os.path.exists(cache_path):
                total += 1
                continue   # already cached

            cam_feat   = extract_cam_feature(backbone, clip_dir, frame_id)
            lidar_feat = extract_lidar_stats(clip_dir, frame_id)

            if cam_feat is None:
                cam_feat = np.zeros(CAM_BACKBONE_DIM, dtype=np.float32)

            np.savez_compressed(cache_path,
                                cam_feat=cam_feat,
                                lidar_feat=lidar_feat)
            total += 1

        print(f"  Cached: {ep_name}/{ev_name}  ({len(ann_files)} frames)")

    print(f"Phase 1 complete — {total} frames cached under {CACHE_ROOT}/\n")


# ─────────────────────────────────────────────
# PHASE 2 — DATASET
# ─────────────────────────────────────────────

class EarlyFusionClip:
    """Holds pre-loaded tensors for one event clip."""
    __slots__ = ("episode", "cam_feats", "lidar_feats", "kin_feats", "labels")

    def __init__(self, episode: int,
                 cam_feats:   np.ndarray,    # (T, 576)
                 lidar_feats: np.ndarray,    # (T, 9)
                 kin_feats:   np.ndarray,    # (T, 19)
                 labels:      np.ndarray):   # (T,)  int
        self.episode    = episode
        self.cam_feats  = cam_feats
        self.lidar_feats= lidar_feats
        self.kin_feats  = kin_feats
        self.labels     = labels


def load_all_clips() -> List[EarlyFusionClip]:
    """
    Load every event clip into memory.
    Returns a list of EarlyFusionClip objects.
    """
    clips_out = []
    total_f = event_f = 0

    for ep_dir in sorted(glob.glob(f"{DATASET_ROOT}/episode_*")):
        ep_idx = int(os.path.basename(ep_dir).split("_")[1])

        for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "event_*"))):
            ev_name   = os.path.basename(ev_dir)
            cache_dir = os.path.join(CACHE_ROOT,
                                     os.path.basename(ep_dir), ev_name)
            ann_files = sorted(glob.glob(os.path.join(ev_dir, "ann", "*.json")))

            if not ann_files:
                continue

            cam_list   = []
            lidar_list = []
            kin_list   = []
            label_list = []
            ok = True

            for ann_path in ann_files:
                frame_id   = os.path.splitext(os.path.basename(ann_path))[0]
                cache_path = os.path.join(cache_dir, f"{frame_id}.npz")

                if not os.path.exists(cache_path):
                    print(f"  WARNING: cache missing for {cache_path} — "
                          f"run without --skip-cache first.")
                    ok = False
                    break

                cached = np.load(cache_path)
                cam_list.append(cached["cam_feat"])       # (576,)
                lidar_list.append(cached["lidar_feat"])   # (9,)

                with open(ann_path) as f:
                    ann = json.load(f)
                kin_list.append(extract_kinematics(ann))  # (19,)
                label_list.append(int(ann["anomaly"]["is_event_frame"]))

            if not ok or len(cam_list) < SEQ_LEN + 1:
                continue

            clip = EarlyFusionClip(
                episode    = ep_idx,
                cam_feats  = np.array(cam_list,   dtype=np.float32),
                lidar_feats= np.array(lidar_list, dtype=np.float32),
                kin_feats  = np.array(kin_list,   dtype=np.float32),
                labels     = np.array(label_list, dtype=np.int64),
            )
            clips_out.append(clip)
            total_f += len(label_list)
            event_f += sum(label_list)

    n_normal = total_f - event_f
    print(f"Loaded {len(clips_out)} clips | {total_f} frames "
          f"({event_f} event / {n_normal} normal = "
          f"{100*event_f/max(total_f,1):.1f}% event)")
    return clips_out


class EarlyFusionDataset(Dataset):
    """
    Sliding-window dataset over EarlyFusionClip objects.

    Each sample is a tuple:
        x_cam   : (SEQ_LEN, 576)  frozen backbone features
        x_kin   : (SEQ_LEN, 19)   kinematic features
        x_lidar : (SEQ_LEN, 9)    LiDAR statistics
        y       : scalar float32  label of the LAST frame in the window
    """

    def __init__(self, clips: List[EarlyFusionClip]):
        self.samples = []

        for clip in clips:
            T = len(clip.labels)
            for t in range(T):
                start = t - SEQ_LEN + 1
                idx   = max(start, 0)
                pad   = max(-start, 0)

                def _window(arr):
                    slc = arr[idx:t+1]                          # real data
                    if pad > 0:
                        slc = np.concatenate(
                            [np.zeros((pad, arr.shape[1]), dtype=np.float32), slc],
                            axis=0
                        )
                    return torch.from_numpy(slc)                # (SEQ_LEN, dim)

                self.samples.append((
                    _window(clip.cam_feats),    # (SEQ_LEN, 576)
                    _window(clip.kin_feats),    # (SEQ_LEN, 19)
                    _window(clip.lidar_feats),  # (SEQ_LEN, 9)
                    torch.tensor(clip.labels[t], dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_loaders(clips: List[EarlyFusionClip]) -> tuple:
    train_clips = [c for c in clips
                   if c.episode not in TEST_EPISODES | VAL_EPISODES]
    val_clips   = [c for c in clips if c.episode in VAL_EPISODES]
    test_clips  = [c for c in clips if c.episode in TEST_EPISODES]

    print(f"\nSplit: train={len(train_clips)} clips | "
          f"val={len(val_clips)} | test={len(test_clips)}")

    # Class weight for weighted BCE (events are majority class)
    all_labels = np.concatenate([c.labels for c in train_clips])
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    pos_w = float(n_neg) / max(float(n_pos), 1.0)
    print(f"pos_weight = {pos_w:.3f}  "
          f"(train: {int(n_pos)} event / {int(n_neg)} normal)")

    train_ds = EarlyFusionDataset(train_clips)
    val_ds   = EarlyFusionDataset(val_clips)
    test_ds  = EarlyFusionDataset(test_clips)

    kw = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw)

    pw = torch.tensor([pos_w], dtype=torch.float32).to(DEVICE)
    return train_loader, val_loader, test_loader, pw


# ─────────────────────────────────────────────
# MODEL — EARLY FUSION BiGRU
# ─────────────────────────────────────────────

class EarlyFusionGRU(nn.Module):
    """
    Early-fusion multi-modal BiGRU for rash-behaviour classification.

    Architecture
    ------------
                 Camera backbone (frozen MobileNetV3-Small, 576-dim)
                         |
              +-----------+-----------+
              |           |           |
         cam_proj     kin_proj   lidar_proj
         576->64       19->32      9->16
              |           |           |
              +-----concat(112)-------+
                          |
                 BiGRU(112, hidden=64, layers=2)  [shared across time]
                          |
                   last hidden (128-dim)
                          |
              Dropout(0.35) -> FC(128,32) -> ReLU
                          |
              Dropout(0.20) -> FC(32, 1)  -> raw logit
    """

    def __init__(self):
        super().__init__()

        # ── Modality projection heads (trained end-to-end) ──
        self.cam_proj = nn.Sequential(
            nn.Linear(CAM_BACKBONE_DIM, CAM_PROJ_DIM),
            nn.LayerNorm(CAM_PROJ_DIM),
            nn.ReLU(),
        )
        self.kin_proj = nn.Sequential(
            nn.Linear(KIN_DIM,       KIN_PROJ_DIM),
            nn.LayerNorm(KIN_PROJ_DIM),
            nn.ReLU(),
        )
        self.lidar_proj = nn.Sequential(
            nn.Linear(LIDAR_STAT_DIM, LIDAR_PROJ_DIM),
            nn.LayerNorm(LIDAR_PROJ_DIM),
            nn.ReLU(),
        )

        # ── Temporal module ──
        self.gru = nn.GRU(
            FUSED_DIM, HIDDEN_DIM,
            num_layers    = N_LAYERS,
            batch_first   = True,
            bidirectional = True,
            dropout       = DROPOUT if N_LAYERS > 1 else 0.0,
        )

        # ── Classification head ──
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM * 2, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self,
                x_cam:   torch.Tensor,   # (B, T, 576)
                x_kin:   torch.Tensor,   # (B, T, 19)
                x_lidar: torch.Tensor,   # (B, T, 9)
                ) -> torch.Tensor:       # (B,) raw logit

        B, T, _ = x_cam.shape

        # Project each modality across the time dimension
        # Reshape to (B*T, dim) for efficient Linear application
        cam_feat   = self.cam_proj  (x_cam.reshape(B*T, -1)).reshape(B, T, -1)
        kin_feat   = self.kin_proj  (x_kin.reshape(B*T, -1)).reshape(B, T, -1)
        lidar_feat = self.lidar_proj(x_lidar.reshape(B*T, -1)).reshape(B, T, -1)

        # Early fusion: concatenate at every timestep (B, T, 112)
        fused = torch.cat([cam_feat, kin_feat, lidar_feat], dim=-1)

        # Temporal reasoning
        gru_out, _ = self.gru(fused)          # (B, T, 128)
        last        = gru_out[:, -1, :]       # (B, 128) — last timestep

        return self.head(last).squeeze(-1)    # (B,)


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimiser=None):
    is_train = optimiser is not None
    model.train(is_train)
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    with torch.set_grad_enabled(is_train):
        for x_cam, x_kin, x_lidar, y in loader:
            x_cam   = x_cam.to(DEVICE)
            x_kin   = x_kin.to(DEVICE)
            x_lidar = x_lidar.to(DEVICE)
            y       = y.to(DEVICE)

            logits = model(x_cam, x_kin, x_lidar)
            loss   = criterion(logits, y)

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            total_loss += loss.item() * len(y)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(y.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return total_loss / max(len(loader.dataset), 1), probs, labels


def metrics(probs, labels, t=0.5):
    preds = (probs >= t).astype(int)
    f1    = f1_score   (labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score   (labels, preds, zero_division=0)
    try:    auc = roc_auc_score(labels, probs)
    except: auc = float("nan")
    return {"f1": f1, "precision": prec, "recall": rec, "auc": auc}


def find_best_threshold(probs, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_confusion(labels, probs, threshold, path):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Rash"],
                yticklabels=["Normal", "Rash"], ax=ax)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix (Early Fusion) — Test Set")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved: {path}")


def plot_roc(labels, probs, auc_val, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"Early Fusion AUROC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Early Fusion) — Test Set")
    ax.legend(); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved: {path}")


def plot_curves(log, path):
    epochs  = [e["epoch"]      for e in log]
    train_f = [e["train_f1"]   for e in log]
    val_f   = [e["val_f1"]     for e in log]
    train_l = [e["train_loss"] for e in log]
    val_l   = [e["val_loss"]   for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, train_l, label="train"); ax1.plot(epochs, val_l, label="val")
    ax1.set_title("Loss (Early Fusion)"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs, train_f, label="train"); ax2.plot(epochs, val_f, label="val")
    ax2.set_title("F1 Score (Early Fusion)"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(skip_cache: bool = False):
    # ── Phase 1: feature extraction ──
    if not skip_cache:
        extract_and_cache_features()
    else:
        print("Skipping feature extraction (--skip-cache set).")

    # ── Load all clips ──
    clips = load_all_clips()
    if not clips:
        print("ERROR: No clips loaded.  Make sure feature cache exists.")
        sys.exit(1)

    train_loader, val_loader, test_loader, pw = make_loaders(clips)

    # ── Model ──
    model     = EarlyFusionGRU().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel : EarlyFusionGRU | Trainable params : {n_params:,}")
    print(f"  Camera branch  : {CAM_BACKBONE_DIM}->{CAM_PROJ_DIM} per frame "
          f"(backbone frozen, projection trained)")
    print(f"  Kinematic branch : {KIN_DIM}->{KIN_PROJ_DIM}")
    print(f"  LiDAR branch   : {LIDAR_STAT_DIM}->{LIDAR_PROJ_DIM}")
    print(f"  Fused per frame: {FUSED_DIM} -> BiGRU(hidden={HIDDEN_DIM}, "
          f"layers={N_LAYERS}, bidirectional)")
    print(f"Training {EPOCHS} epochs on {DEVICE}\n")

    best_val_f1 = -1.0
    ckpt_path   = os.path.join(MODEL_DIR, "early_fusion_best.pt")
    log         = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_probs, tr_labels = run_epoch(
            model, train_loader, criterion, optimiser)
        vl_loss, vl_probs, vl_labels = run_epoch(
            model, val_loader, criterion)
        scheduler.step()

        tr_m = metrics(tr_probs, tr_labels)
        vl_m = metrics(vl_probs, vl_labels)

        log.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 5),
            "train_f1":   round(tr_m["f1"], 4),
            "val_loss":   round(vl_loss, 5),
            "val_f1":     round(vl_m["f1"], 4),
            "val_auc":    round(vl_m["auc"], 4),
        })

        if vl_m["f1"] > best_val_f1:
            best_val_f1 = vl_m["f1"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      best_val_f1,
            }, ckpt_path)
            star = " << best"
        else:
            star = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train: loss={tr_loss:.4f} F1={tr_m['f1']:.3f}  |  "
                  f"val: loss={vl_loss:.4f} F1={vl_m['f1']:.3f} "
                  f"AUC={vl_m['auc']:.3f}{star}")

    print(f"\nBest val F1 : {best_val_f1:.4f}  (checkpoint: {ckpt_path})")

    with open(os.path.join(MODEL_DIR, "early_fusion_training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    plot_curves(log, os.path.join(MODEL_DIR, "early_fusion_training_curves.png"))

    # ── Test evaluation ──
    print("\n--- Test set evaluation ---")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, vl_probs, vl_labels = run_epoch(model, val_loader,  criterion)
    _, ts_probs, ts_labels = run_epoch(model, test_loader, criterion)

    best_t, _ = find_best_threshold(vl_probs, vl_labels)
    print(f"Optimal threshold (from val set): {best_t:.2f}")

    if ts_labels.sum() == 0:
        print("WARNING: No event frames in test set. Adjust TEST_EPISODES.")
    else:
        ts_m = metrics(ts_probs, ts_labels, t=best_t)
        print(f"\n  F1        : {ts_m['f1']:.4f}")
        print(f"  Precision : {ts_m['precision']:.4f}")
        print(f"  Recall    : {ts_m['recall']:.4f}")
        print(f"  AUROC     : {ts_m['auc']:.4f}")
        print()
        preds = (ts_probs >= best_t).astype(int)
        print(classification_report(ts_labels, preds,
                                    target_names=["Normal", "Rash Event"],
                                    zero_division=0))

        plot_confusion(ts_labels, ts_probs, best_t,
                       os.path.join(MODEL_DIR, "early_fusion_confusion_matrix.png"))
        try:
            plot_roc(ts_labels, ts_probs, ts_m["auc"],
                     os.path.join(MODEL_DIR, "early_fusion_roc_curve.png"))
        except Exception as e:
            print(f"ROC skipped: {e}")

    # ── Side-by-side summary vs kinematic-only ──
    print("\n--- Model comparison ---")
    print("  (kinematic-only results are from best checkpoint in models/)")
    kin_ckpt_path = os.path.join(MODEL_DIR, "kinematic_gru_best.pt")
    if os.path.exists(kin_ckpt_path):
        kin_ckpt = torch.load(kin_ckpt_path, map_location=DEVICE, weights_only=False)
        print(f"  Kinematic-only  val F1 = {kin_ckpt['val_f1']:.4f}")
    print(f"  Early Fusion    val F1 = {best_val_f1:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip feature extraction and use existing cache")
    args = parser.parse_args()
    main(skip_cache=args.skip_cache)
