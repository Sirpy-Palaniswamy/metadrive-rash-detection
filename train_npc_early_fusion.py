"""
NPC Rash-Behaviour Early-Fusion Classifier
==========================================
True early fusion: all four modalities are processed into a common
spatial representation and concatenated as channels before any task-specific
learning begins.

Data sources used
-----------------
  JSON  (ego + NPC kinematics)
  RGB   (front camera)
  Semantic  (front camera)
  LiDAR (intensity statistics — see note below)

LiDAR BEV note
--------------
The current dataset was collected before the LiDAR coordinate bug was fixed
in collect_dataset.py v3.  The raw X/Y/Z values are in world frame (span
±1 000 m) rather than ego-centric ±100 m, so they cannot be used for proper
BEV projection.  Intensity IS correct (formula: 1 − dist/100).

Work-around (used here):
  * A 2-channel "scene BEV" is synthesised from the JSON annotation NPC
    positions (which are accurately ego-centric) and placed in channels 0-1.
  * 9 LiDAR intensity statistics are appended to the tabular branch.

When new data is collected with the fixed collector the two JSON-BEV
channels can be replaced with 3-channel LiDAR BEV without changing the rest
of the architecture (just set SPATIAL_C = 3+3+1 = 7 and swap in lidar_to_bev).

Architecture
------------
  Spatial input  (6, 112, 112):
    Scene BEV  (ch 0-1): NPC occupancy | NPC speed              (from JSON)
    Front RGB  (ch 2-4): ImageNet-normalised, resized 640×360
    Front Sem  (ch 5)  : palette index / 255, resized 640×360

  CNN  (6→128 spatial features)
    Block1: Conv(6,32)×2 + BN + ReLU  → MaxPool  →  32×56×56
    Block2: Conv(32,64)×2 + BN + ReLU → MaxPool  →  64×28×28
    Block3: Conv(64,128)×2+ BN + ReLU → GAP(4×4) → Flatten → FC(2048,128)

  Tabular input  (23-dim):
    Ego kinematics:      speed, sin(hdg), cos(hdg)                 — 3
    Target NPC:          dx, dy, speed, sin(hdg), cos(hdg),
                         distance, length, width, height, vx, vy   — 11
    LiDAR intensity stats: n_pts, mean, std, p10, p25, p75, p90,
                           frac_high, z_std                        — 9
    MLP: FC(23,64)+LN+ReLU → FC(64,32)+ReLU                       → 32

  Fusion head:
    Concat(128+32) → FC(64)+ReLU+Drop → FC(1) → logit

Usage
-----
  python train_npc_early_fusion.py               # full run
  python train_npc_early_fusion.py --skip-cache  # skip BEV pre-computation
  python train_npc_early_fusion.py --epochs 80

Output (in models/)
------
  npc_ef_best.pt, npc_ef_training_log.json
  npc_ef_training_curves.png, npc_ef_confusion_matrix.png, npc_ef_roc_curve.png
"""

import argparse, json, os, glob, math, random, sys, warnings
from typing import List, Optional
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
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


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DATASET_ROOT   = "metadrive_fusion_dataset"
BEV_CACHE_ROOT = "scene_bev_cache"   # cache for JSON-derived scene BEV
MODEL_DIR      = "models"
os.makedirs(MODEL_DIR,      exist_ok=True)
os.makedirs(BEV_CACHE_ROOT, exist_ok=True)

# Scene BEV parameters (ego-centric top-down view, from JSON annotations)
BEV_RANGE = 50.0          # ±50 m
BEV_RES   = 0.5           # metres / pixel
BEV_SIZE  = int(2 * BEV_RANGE / BEV_RES)   # 200 × 200
BEV_C     = 2             # occupancy | NPC speed

# Image inputs
IMG_SIZE  = 112            # all spatial inputs are resized to this
RGB_C     = 3
SEM_C     = 1

# Combined spatial input: scene-BEV(2) + RGB(3) + Sem(1) = 6 channels
SPATIAL_C = BEV_C + RGB_C + SEM_C   # 6

# Tabular: ego(3) + NPC(11) + LiDAR-stats(9) = 23
N_LIDAR_STATS = 9
TABULAR_DIM   = 3 + 11 + N_LIDAR_STATS   # 23

MAX_DIST  = 50.0           # only predict for NPCs within this distance (m)
FRONT_CAM = "front"

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Training
BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.4
SEED         = 42

# Episode-level splits (consistent with existing training scripts)
TEST_EPISODES = {9, 10}
VAL_EPISODES  = {3, 6}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ─── SCENE BEV FROM JSON ANNOTATIONS ─────────────────────────────────────────

def generate_scene_bev(ann: dict) -> np.ndarray:
    """
    Synthesise a 2-channel ego-centric BEV from annotation NPC positions.

    This serves the same role as a LiDAR BEV for the current dataset because
    the raw LiDAR XYZ coordinates are in world frame (a known collection bug).
    The JSON position_ego fields are correctly ego-centric.

    BEV convention:
      col = (x_ego + BEV_RANGE) / BEV_RES   (x+ = right)
      row = (BEV_RANGE - y_ego) / BEV_RES   (y+ = forward = top of image)

    Output channels (2, BEV_SIZE, BEV_SIZE):
      0  occupancy  — 1.0 at each NPC centre pixel, 0 elsewhere
      1  speed      — NPC speed / 30 at each occupied pixel, 0 elsewhere
    """
    bev = np.zeros((2, BEV_SIZE, BEV_SIZE), dtype=np.float32)
    for obj in ann.get("objects", []):
        if obj.get("class") != "vehicle":
            continue
        pos = obj.get("position_ego", [0.0, 0.0, 0.0])
        dx, dy = float(pos[0]), float(pos[1])
        if abs(dx) >= BEV_RANGE or abs(dy) >= BEV_RANGE:
            continue
        col = int((dx + BEV_RANGE) / BEV_RES)
        row = int((BEV_RANGE - dy) / BEV_RES)
        col = max(0, min(col, BEV_SIZE - 1))
        row = max(0, min(row, BEV_SIZE - 1))

        dims = obj.get("dimensions_lwh", [4.0, 2.0, 1.7])
        length_px = max(1, int(float(dims[0]) / BEV_RES))
        width_px  = max(1, int(float(dims[1]) / BEV_RES))
        r0 = max(0, row - length_px // 2)
        r1 = min(BEV_SIZE, row + length_px // 2 + 1)
        c0 = max(0, col - width_px  // 2)
        c1 = min(BEV_SIZE, col + width_px  // 2 + 1)

        bev[0, r0:r1, c0:c1] = 1.0
        bev[1, r0:r1, c0:c1] = float(obj.get("speed_ms", 0.0)) / 30.0

    return bev


def _bev_cache_path(ep_name: str, ev_name: str, frame_id: str) -> str:
    return os.path.join(BEV_CACHE_ROOT, ep_name, ev_name, f"{frame_id}.npy")


def load_or_cache_bev(ep_name: str, ev_name: str,
                      frame_id: str, ann: dict) -> np.ndarray:
    """Return cached scene BEV, computing and storing it first if absent."""
    path = _bev_cache_path(ep_name, ev_name, frame_id)
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bev = generate_scene_bev(ann)
    np.save(path, bev)
    return bev


def precompute_bev_cache() -> None:
    """One-time pass: compute and cache scene BEVs for all frames (~30 s)."""
    print("\n=== Pre-computing scene BEV cache ===")
    total = 0
    for ep_dir in sorted(glob.glob(f"{DATASET_ROOT}/episode_*")):
        ep_name = os.path.basename(ep_dir)
        for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "event_*"))):
            ev_name   = os.path.basename(ev_dir)
            ann_files = sorted(glob.glob(os.path.join(ev_dir, "ann", "*.json")))
            for ann_path in ann_files:
                frame_id = os.path.splitext(os.path.basename(ann_path))[0]
                if not os.path.exists(_bev_cache_path(ep_name, ev_name, frame_id)):
                    with open(ann_path) as f:
                        ann = json.load(f)
                    bev = generate_scene_bev(ann)
                    path = _bev_cache_path(ep_name, ev_name, frame_id)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    np.save(path, bev)
                total += 1
            print(f"  {ep_name}/{ev_name}: {len(ann_files)} frames")
    print(f"Scene BEV cache complete — {total} frames\n")


# ─── LIDAR INTENSITY STATISTICS ───────────────────────────────────────────────

def extract_lidar_stats(clip_dir: str, frame_id: str) -> np.ndarray:
    """
    Compute 9 intensity-based statistics from the LiDAR point cloud.

    The LiDAR XYZ coordinates in the current dataset are in world frame
    (a known collection bug).  Intensity IS correct: 1 − dist/100.

    Features (9-dim):
      0  n_points_norm        point count / 30 000
      1  intensity_mean
      2  intensity_std
      3  intensity_p10        10th percentile
      4  intensity_p25        25th percentile
      5  intensity_p75        75th percentile
      6  intensity_p90        90th percentile
      7  frac_high_intensity  fraction with intensity > 0.6  (nearby objects)
      8  z_std_norm           std(Z) / 20  (Z is less affected by the bug)
    """
    stats = np.zeros(N_LIDAR_STATS, dtype=np.float32)
    lidar_path = os.path.join(clip_dir, "lidar", f"{frame_id}.npy")
    if not os.path.exists(lidar_path):
        return stats
    pts = np.load(lidar_path)
    if pts.ndim != 2 or pts.shape[1] < 4 or pts.shape[0] == 0:
        return stats
    intensity = pts[:, 3].astype(np.float32)
    z_vals    = pts[:, 2].astype(np.float32)
    stats[0]  = min(len(pts) / 30000.0, 1.0)
    stats[1]  = float(intensity.mean())
    stats[2]  = float(intensity.std())
    stats[3]  = float(np.percentile(intensity, 10))
    stats[4]  = float(np.percentile(intensity, 25))
    stats[5]  = float(np.percentile(intensity, 75))
    stats[6]  = float(np.percentile(intensity, 90))
    stats[7]  = float((intensity > 0.6).mean())
    stats[8]  = float(np.clip(z_vals.std() / 20.0, 0.0, 1.0))
    return stats


# ─── TABULAR FEATURE EXTRACTION ───────────────────────────────────────────────

def extract_tabular(ego: dict, obj: dict,
                    lidar_stats: np.ndarray) -> np.ndarray:
    """
    Build a 23-dim float32 vector for one (ego, target-NPC) pair.

    Ego kinematics (3):
      [0]  ego speed / 30
      [1]  sin(ego heading)
      [2]  cos(ego heading)

    Target NPC kinematics (11):
      [3]  dx_ego / 50       [4]  dy_ego / 50
      [5]  NPC speed / 30
      [6]  sin(heading_ego)  [7]  cos(heading_ego)
      [8]  distance / 50
      [9]  length / 5        [10] width / 5    [11] height / 5
      [12] vx_world / 15     [13] vy_world / 15

    LiDAR intensity statistics (9):
      [14..22]  see extract_lidar_stats()
    """
    feat = np.zeros(TABULAR_DIM, dtype=np.float32)

    eg_spd  = float(ego.get("speed_ms", 0.0))
    eg_head = float(ego.get("heading_rad", 0.0))
    feat[0] = eg_spd / 30.0
    feat[1] = math.sin(eg_head)
    feat[2] = math.cos(eg_head)

    pos = obj.get("position_ego", [0.0, 0.0, 0.0])
    feat[3] = float(pos[0]) / 50.0
    feat[4] = float(pos[1]) / 50.0
    feat[5] = float(obj.get("speed_ms", 0.0)) / 30.0
    hdg = float(obj.get("heading_rad_ego", 0.0))
    feat[6] = math.sin(hdg)
    feat[7] = math.cos(hdg)
    feat[8] = float(obj.get("distance_to_ego", 0.0)) / 50.0
    dims = obj.get("dimensions_lwh", [4.0, 2.0, 1.7])
    feat[9]  = float(dims[0]) / 5.0
    feat[10] = float(dims[1]) / 5.0
    feat[11] = float(dims[2]) / 5.0
    vel = obj.get("velocity", [0.0, 0.0, 0.0])
    feat[12] = float(vel[0]) / 15.0
    feat[13] = float(vel[1]) / 15.0

    feat[14:23] = lidar_stats

    return feat


# ─── IMAGE LOADING ────────────────────────────────────────────────────────────

def _load_rgb(path: str) -> np.ndarray:
    """(3, IMG_SIZE, IMG_SIZE) float32, ImageNet-normalised."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    arr = img_rgb.astype(np.float32) / 255.0   # (H, W, 3)
    arr = arr.transpose(2, 0, 1)               # (3, H, W)
    return (arr - IMG_MEAN) / IMG_STD


def _load_semantic(path: str) -> np.ndarray:
    """(1, IMG_SIZE, IMG_SIZE) float32 — class index / 255.
    Uses PIL to correctly decode palette-PNG class indices."""
    if not os.path.exists(path):
        return np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    pil_img = Image.open(path)
    arr = np.array(pil_img if pil_img.mode == "P"
                   else pil_img.convert("L"), dtype=np.uint8)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE),
                     interpolation=cv2.INTER_NEAREST)
    return (arr.astype(np.float32) / 255.0)[np.newaxis]   # (1, H, W)


# ─── DATASET ──────────────────────────────────────────────────────────────────

class NPCRashDataset(Dataset):
    """
    Per-NPC rash-behaviour dataset.

    Each sample = (frame, NPC) for every vehicle within MAX_DIST metres.
    Label: obj["is_rogue"]  (1 = rash/rogue, 0 = normal)

    __getitem__ returns:
        spatial  (SPATIAL_C, IMG_SIZE, IMG_SIZE)  — early-fused tensor
        tabular  (TABULAR_DIM,)                    — kinematic + LiDAR stats
        label    scalar float32
    """

    def __init__(self, frame_records: List[dict]) -> None:
        self.samples: List[dict] = []
        n_pos = n_neg = 0

        for rec in frame_records:
            with open(rec["ann_path"]) as f:
                ann = json.load(f)

            ego  = ann["ego"]
            npcs = [o for o in ann.get("objects", [])
                    if o.get("class") == "vehicle"
                    and float(o.get("distance_to_ego", 9999)) <= MAX_DIST]

            for obj in npcs:
                label = int(bool(obj.get("is_rogue", False)))
                self.samples.append({
                    "ep_name":  rec["ep_name"],
                    "ev_name":  rec["ev_name"],
                    "frame_id": rec["frame_id"],
                    "clip_dir": rec["clip_dir"],
                    "ann":      ann,
                    "ego":      ego,
                    "obj":      obj,
                    "label":    label,
                })
                if label == 1:
                    n_pos += 1
                else:
                    n_neg += 1

        n_total = n_pos + n_neg
        pct = 100.0 * n_pos / max(n_total, 1)
        print(f"    {n_total} NPC samples "
              f"({n_pos} rogue / {n_neg} normal = {pct:.1f}% rogue)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s        = self.samples[idx]
        frame_id = s["frame_id"]
        clip_dir = s["clip_dir"]

        # ── Scene BEV  (2, BEV_SIZE, BEV_SIZE) → (2, IMG_SIZE, IMG_SIZE) ──
        bev = load_or_cache_bev(s["ep_name"], s["ev_name"], frame_id, s["ann"])
        bev_resized = np.stack([
            cv2.resize(bev[c], (IMG_SIZE, IMG_SIZE),
                       interpolation=cv2.INTER_LINEAR)
            for c in range(BEV_C)
        ], axis=0)                                           # (2, H, W)

        # ── Front-camera RGB  (3, IMG_SIZE, IMG_SIZE) ──
        rgb = _load_rgb(os.path.join(clip_dir, "rgb",
                                     f"{frame_id}_{FRONT_CAM}.png"))

        # ── Front-camera Semantic  (1, IMG_SIZE, IMG_SIZE) ──
        sem = _load_semantic(os.path.join(clip_dir, "semantic",
                                          f"{frame_id}_{FRONT_CAM}.png"))

        # ── Early spatial fusion: concatenate all channels ──────────────────
        # (2 scene-BEV) + (3 RGB) + (1 Sem) = 6 channels
        spatial = np.concatenate([bev_resized, rgb, sem], axis=0).astype(np.float32)

        # ── LiDAR intensity stats (added to tabular) ──
        lidar_stats = extract_lidar_stats(clip_dir, frame_id)

        # ── Tabular: ego + NPC kinematics + LiDAR stats ──
        tabular = extract_tabular(s["ego"], s["obj"], lidar_stats)

        return (
            torch.from_numpy(spatial),
            torch.from_numpy(tabular),
            torch.tensor(s["label"], dtype=torch.float32),
        )


# ─── DATASET HELPERS ──────────────────────────────────────────────────────────

def scan_dataset() -> List[dict]:
    """Return one record per annotation frame across the entire dataset."""
    records = []
    for ep_dir in sorted(glob.glob(f"{DATASET_ROOT}/episode_*")):
        ep_name = os.path.basename(ep_dir)
        ep_idx  = int(ep_name.split("_")[1])
        for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "event_*"))):
            ev_name = os.path.basename(ev_dir)
            for ann_path in sorted(
                    glob.glob(os.path.join(ev_dir, "ann", "*.json"))):
                frame_id = os.path.splitext(os.path.basename(ann_path))[0]
                records.append({
                    "ep_name":  ep_name,
                    "ev_name":  ev_name,
                    "ep_idx":   ep_idx,
                    "frame_id": frame_id,
                    "ann_path": ann_path,
                    "clip_dir": ev_dir,
                })
    return records


def make_loaders(records: List[dict]):
    train_recs = [r for r in records
                  if r["ep_idx"] not in TEST_EPISODES | VAL_EPISODES]
    val_recs   = [r for r in records if r["ep_idx"] in VAL_EPISODES]
    test_recs  = [r for r in records if r["ep_idx"] in TEST_EPISODES]

    n_ep = lambda rs: len({r["ep_idx"] for r in rs})
    print(f"\nEpisode split: train={n_ep(train_recs)} | "
          f"val={n_ep(val_recs)} | test={n_ep(test_recs)}")

    print("Building datasets:")
    print("  Train:", end=" "); train_ds = NPCRashDataset(train_recs)
    print("  Val:  ", end=" "); val_ds   = NPCRashDataset(val_recs)
    print("  Test: ", end=" "); test_ds  = NPCRashDataset(test_recs)

    n_pos = int(sum(s["label"] for s in train_ds.samples))
    n_neg = len(train_ds.samples) - n_pos
    pos_w = float(n_neg) / max(float(n_pos), 1.0)
    print(f"\npos_weight = {pos_w:.2f}  (train: {n_pos} rogue / {n_neg} normal)")

    kw = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, **kw)

    pw = torch.tensor([pos_w], dtype=torch.float32).to(DEVICE)
    return train_loader, val_loader, test_loader, pw


# ─── MODEL ────────────────────────────────────────────────────────────────────

class EarlyFusionNPCClassifier(nn.Module):
    """
    Early-fusion multi-modal classifier for per-NPC rash-behaviour prediction.

    All spatial modalities (scene BEV, RGB, Semantic) are concatenated at the
    channel dimension BEFORE any learned processing begins — this is early
    (input-level) fusion.  The CNN therefore learns cross-modal spatial
    correlations from the very first convolutional layer.

    LiDAR is represented by 9 intensity statistics in the tabular branch,
    combined with ego and NPC kinematic features.
    """

    def __init__(self, tabular_dim: int = TABULAR_DIM,
                 dropout: float = DROPOUT) -> None:
        super().__init__()

        # ── Spatial CNN  (early fusion of all SPATIAL_C channels) ────────────
        self.spatial_cnn = nn.Sequential(
            # Block 1: (SPATIAL_C, 112, 112) → (32, 56, 56)
            nn.Conv2d(SPATIAL_C, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: (32, 56, 56) → (64, 28, 28)
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: (64, 28, 28) → (128, 4, 4)
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),                            # → 2048
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )                                            # output: (B, 128)

        # ── Tabular MLP  (kinematics + LiDAR stats) ──────────────────────────
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )                                            # output: (B, 32)

        # ── Fusion head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )                                            # output: (B, 1)

    def forward(self, spatial: torch.Tensor,
                tabular: torch.Tensor) -> torch.Tensor:
        sp  = self.spatial_cnn(spatial)              # (B, 128)
        tab = self.tabular_mlp(tabular)              # (B, 32)
        return self.head(torch.cat([sp, tab], dim=-1)).squeeze(-1)   # (B,)


# ─── TRAINING UTILITIES ───────────────────────────────────────────────────────

def run_epoch(model: nn.Module, loader: DataLoader,
              criterion: nn.Module, optimiser=None) -> tuple:
    is_train = optimiser is not None
    model.train(is_train)
    total_loss = 0.0
    all_probs:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.set_grad_enabled(is_train):
        for spatial, tabular, y in loader:
            spatial = spatial.to(DEVICE, non_blocking=True)
            tabular = tabular.to(DEVICE, non_blocking=True)
            y       = y.to(DEVICE, non_blocking=True)

            logits = model(spatial, tabular)
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


def compute_metrics(probs: np.ndarray, labels: np.ndarray,
                    threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    f1   = f1_score        (labels, preds, zero_division=0)
    prec = precision_score (labels, preds, zero_division=0)
    rec  = recall_score    (labels, preds, zero_division=0)
    try:    auc = roc_auc_score(labels, probs)
    except: auc = float("nan")
    return {"f1": f1, "precision": prec, "recall": rec, "auc": auc}


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# ─── VISUALISATION ────────────────────────────────────────────────────────────

def plot_confusion(labels, probs, threshold, path):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Rogue"],
                yticklabels=["Normal", "Rogue"], ax=ax)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix — NPC Early Fusion (Test)")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved: {path}")


def plot_roc(labels, probs, auc_val, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve — NPC Early Fusion (Test)")
    ax.legend(); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved: {path}")


def plot_curves(log: list, path: str) -> None:
    epochs  = [e["epoch"]      for e in log]
    train_l = [e["train_loss"] for e in log]
    val_l   = [e["val_loss"]   for e in log]
    train_f = [e["train_f1"]   for e in log]
    val_f   = [e["val_f1"]     for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, train_l, label="train"); ax1.plot(epochs, val_l, label="val")
    ax1.set_title("BCE Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs, train_f, label="train"); ax2.plot(epochs, val_f, label="val")
    ax2.set_title("F1 Score"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(skip_cache: bool = False, epochs: int = EPOCHS) -> None:

    # Phase 1: scene BEV cache
    if not skip_cache:
        precompute_bev_cache()
    else:
        print("Skipping BEV pre-computation (--skip-cache).")

    records = scan_dataset()
    print(f"\nTotal frames scanned: {len(records)}")
    train_loader, val_loader, test_loader, pw = make_loaders(records)

    if len(train_loader.dataset) == 0:
        print("ERROR: empty training set."); sys.exit(1)

    model     = EarlyFusionNPCClassifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                            T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EarlyFusionNPCClassifier")
    print(f"  Trainable parameters : {n_params:,}")
    print(f"  Spatial input        : ({SPATIAL_C}, {IMG_SIZE}, {IMG_SIZE})")
    print(f"    Scene BEV {BEV_C}ch  |  Front RGB {RGB_C}ch  |  "
          f"Front Semantic {SEM_C}ch")
    print(f"  Tabular input        : {TABULAR_DIM}-dim  "
          f"(ego + NPC kinematics + {N_LIDAR_STATS} LiDAR stats)")
    print(f"Training {epochs} epochs  |  batch={BATCH_SIZE}  "
          f"|  lr={LR}  |  device={DEVICE}\n")

    ckpt_path   = os.path.join(MODEL_DIR, "npc_ef_best.pt")
    best_val_f1 = -1.0
    log         = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_probs, tr_labels = run_epoch(
            model, train_loader, criterion, optimiser)
        vl_loss, vl_probs, vl_labels = run_epoch(
            model, val_loader, criterion)
        scheduler.step()

        tr_m = compute_metrics(tr_probs, tr_labels)
        vl_m = compute_metrics(vl_probs, vl_labels)

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
                "config": {
                    "SPATIAL_C":    SPATIAL_C,
                    "IMG_SIZE":     IMG_SIZE,
                    "TABULAR_DIM":  TABULAR_DIM,
                    "BEV_RANGE":    BEV_RANGE,
                    "BEV_RES":      BEV_RES,
                    "DROPOUT":      DROPOUT,
                },
            }, ckpt_path)
            star = " << best"
        else:
            star = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train: loss={tr_loss:.4f} F1={tr_m['f1']:.3f}  |  "
                  f"val: loss={vl_loss:.4f} F1={vl_m['f1']:.3f} "
                  f"AUC={vl_m['auc']:.3f}{star}")

    print(f"\nBest val F1: {best_val_f1:.4f}  →  {ckpt_path}")

    log_path    = os.path.join(MODEL_DIR, "npc_ef_training_log.json")
    curves_path = os.path.join(MODEL_DIR, "npc_ef_training_curves.png")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    plot_curves(log, curves_path)

    # ── Test set evaluation ──────────────────────────────────────────────────
    print("\n--- Test set evaluation ---")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, vl_probs, vl_labels = run_epoch(model, val_loader,  criterion)
    _, ts_probs, ts_labels = run_epoch(model, test_loader, criterion)

    best_t, _ = find_best_threshold(vl_probs, vl_labels)
    print(f"Optimal threshold (calibrated on val set): {best_t:.2f}")

    if ts_labels.sum() == 0:
        print("WARNING: no rogue NPCs in test episodes "
              f"{sorted(TEST_EPISODES)} — consider adjusting TEST_EPISODES.")
    else:
        ts_m = compute_metrics(ts_probs, ts_labels, threshold=best_t)
        print(f"\n  F1        : {ts_m['f1']:.4f}")
        print(f"  Precision : {ts_m['precision']:.4f}")
        print(f"  Recall    : {ts_m['recall']:.4f}")
        print(f"  AUROC     : {ts_m['auc']:.4f}\n")
        preds = (ts_probs >= best_t).astype(int)
        print(classification_report(ts_labels, preds,
                                    target_names=["Normal", "Rogue"],
                                    zero_division=0))
        plot_confusion(ts_labels, ts_probs, best_t,
                       os.path.join(MODEL_DIR, "npc_ef_confusion_matrix.png"))
        try:
            plot_roc(ts_labels, ts_probs, ts_m["auc"],
                     os.path.join(MODEL_DIR, "npc_ef_roc_curve.png"))
        except Exception as exc:
            print(f"ROC skipped: {exc}")

    # ── Comparison ──────────────────────────────────────────────────────────
    print("\n--- Model comparison ---")
    prev = os.path.join(MODEL_DIR, "early_fusion_best.pt")
    if os.path.exists(prev):
        prev_ckpt = torch.load(prev, map_location="cpu", weights_only=False)
        print(f"  Frame-level early fusion  val F1 : {prev_ckpt['val_f1']:.4f}")
    print(f"  NPC-level  early fusion   val F1 : {best_val_f1:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train per-NPC early-fusion rash-behaviour classifier")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip BEV pre-computation and use existing cache")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    args = parser.parse_args()
    main(skip_cache=args.skip_cache, epochs=args.epochs)
