"""
Early-Fusion Multi-Modal Rash-Behaviour Classifier  (v2 — Full)
================================================================
Fuses FOUR modalities at the frame level before temporal reasoning:

  1. LiDAR BEV  -- 360-deg point cloud projected to a 64x64 top-down grid
                   3 channels: occupancy, mean intensity, mean height
                   Processed by a trainable lightweight CNN -> 64-dim

  2. RGB Camera -- 6 surround-view RGB images (640x360)
                   Frozen MobileNetV3-Small backbone -> mean-pool across
                   6 views -> trainable Linear projection -> 64-dim

  3. Semantic   -- 6 surround-view semantic segmentation images
                   Same frozen backbone -> separate projection -> 32-dim

  4. JSON / Kinematics
                -- The numerical motion data from annotation JSON files.
                   "Kinematics" is NOT a separate sensor; it is simply the
                   numbers already stored in every annotation file:
                     ego speed, ego heading, nearby-vehicle positions,
                     speeds, headings, and distances.
                   These are arranged into a fixed 19-dim vector -> 32-dim

Early Fusion
  per-frame fused feature = concat(lidar_64, rgb_64, sem_32, json_32) = 192-dim
  fed as a TIME SEQUENCE into a Bidirectional GRU

BEV specification
  Range:  X in [-32, +32] m  (lateral)
          Y in [-16, +48] m  (mostly forward: cut-ins happen in front)
  Grid:   64 x 64 pixels at 1 m/pixel
  Origin: pixel (32, 16) = ego vehicle position

Three-phase execution
  Phase 1a -- extract & cache frozen backbone features (RGB + Semantic)
  Phase 1b -- generate & cache BEV grids from LiDAR .npy files
  Phase 2  -- train the full model end-to-end (BEV CNN trainable)

Usage
-----
  python train_early_fusion_v2.py            # full run
  python train_early_fusion_v2.py --skip-cache   # skip phases 1a/1b

Output
------
  models/ef2_best.pt
  models/ef2_training_log.json
  models/ef2_training_curves.png
  models/ef2_confusion_matrix.png
  models/ef2_roc_curve.png
"""

import argparse, json, os, glob, math, random, sys, time, warnings
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_kinematic_lstm import extract_frame_features as extract_json_features
from train_kinematic_lstm import FEAT_DIM as JSON_DIM   # 19

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_ROOT = "metadrive_fusion_dataset"
CACHE_ROOT   = "feature_cache_v2"
MODEL_DIR    = "models"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

CAMERA_NAMES = ["front", "front_left", "front_right",
                "back",  "back_left",  "back_right"]

# ── BEV grid ──
BEV_X_MIN, BEV_X_MAX = -32.0, 32.0   # lateral metres
BEV_Y_MIN, BEV_Y_MAX = -16.0, 48.0   # longitudinal metres (mostly forward)
BEV_RES              =  1.0           # metres per pixel
BEV_W = int((BEV_X_MAX - BEV_X_MIN) / BEV_RES)   # 64
BEV_H = int((BEV_Y_MAX - BEV_Y_MIN) / BEV_RES)   # 64
BEV_EGO_COL = int((0.0 - BEV_X_MIN) / BEV_RES)   # 32 (centre column)
BEV_EGO_ROW = int((0.0 - BEV_Y_MIN) / BEV_RES)   # 16 (lower quarter)

# ── Feature dims ──
CAM_BB_DIM   = 576    # MobileNetV3-Small pool output (frozen)
RGB_PROJ_DIM = 64     # trainable projection
SEM_PROJ_DIM = 32
LIDAR_BEV_CH = 3      # occupancy, intensity, height
LIDAR_PROJ   = 64
JSON_PROJ    = 32
FUSED_DIM    = RGB_PROJ_DIM + SEM_PROJ_DIM + LIDAR_PROJ + JSON_PROJ  # 192

# ── Temporal model ──
SEQ_LEN    = 10
HIDDEN_DIM = 96
N_LAYERS   = 2
DROPOUT    = 0.50          # increased from 0.35 to fight BEV-encoder overfitting

# ── Training ──
BATCH_SIZE     = 32
EPOCHS         = 40        # fewer; best is always early; plateau ~ep5-10
LR             = 3e-4
WEIGHT_DECAY   = 5e-4      # stronger L2 (was 1e-4)
PATIENCE       = 12        # early-stop if val F1 doesn't improve for 12 epochs

# ── Episode splits ──
TEST_EPISODES = {9, 10, 47, 49}      # held-out for final evaluation
VAL_EPISODES  = {3, 6, 45, 46}       # used for val-F1 checkpoint selection

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
print(f"BEV grid: {BEV_W}x{BEV_H} px, {BEV_RES}m/px, "
      f"X[{BEV_X_MIN},{BEV_X_MAX}]m  Y[{BEV_Y_MIN},{BEV_Y_MAX}]m")

IMG_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# PHASE 1a — BACKBONE FEATURE EXTRACTION
# ─────────────────────────────────────────────

def build_backbone() -> nn.Module:
    model  = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
    return encoder.eval().to(DEVICE)


def _load_view_tensors(clip_dir: str, frame_id: str,
                       subfolder: str) -> List[torch.Tensor]:
    """Load up to 6 camera images as pre-processed tensors (no backbone yet)."""
    tensors = []
    for cam in CAMERA_NAMES:
        path = os.path.join(clip_dir, subfolder, f"{frame_id}_{cam}.png")
        if not os.path.exists(path):
            continue
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensors.append(IMG_TRANSFORM(img_rgb))
    return tensors


def extract_both_modalities(backbone: nn.Module, clip_dir: str,
                            frame_id: str) -> Tuple[Optional[np.ndarray],
                                                    Optional[np.ndarray]]:
    """
    Extract RGB and Semantic features in ONE batched backbone forward pass.
    Stacks up to 12 images (6 RGB + 6 semantic), runs backbone once,
    then splits and mean-pools each modality.

    This is ~6-12x faster than sequential single-image calls.
    Returns (rgb_feat (576,), sem_feat (576,)) -- either may be None.
    """
    rgb_tensors = _load_view_tensors(clip_dir, frame_id, "rgb")
    sem_tensors = _load_view_tensors(clip_dir, frame_id, "semantic")

    if not rgb_tensors and not sem_tensors:
        return None, None

    n_rgb = len(rgb_tensors)
    n_sem = len(sem_tensors)

    # Build combined batch: [rgb0..rgbN, sem0..semM]
    all_tensors = rgb_tensors + sem_tensors   # each is (3, 224, 224)
    batch = torch.stack(all_tensors).to(DEVICE)  # (N+M, 3, 224, 224)

    with torch.no_grad():
        all_feats = backbone(batch).cpu().numpy()  # (N+M, 576)

    rgb_feat = all_feats[:n_rgb].mean(0).astype(np.float32) if n_rgb else None
    sem_feat = all_feats[n_rgb:].mean(0).astype(np.float32) if n_sem else None
    return rgb_feat, sem_feat


# ─────────────────────────────────────────────
# PHASE 1b — BEV GRID GENERATION
# ─────────────────────────────────────────────

def lidar_to_bev(npy_path: str) -> np.ndarray:
    """
    Convert a LiDAR .npy point cloud (N, 4) [X, Y, Z, intensity]
    into a 3-channel BEV grid (3, BEV_H, BEV_W).

    Channel 0: binary occupancy  (any point in cell -> 1)
    Channel 1: mean intensity    (0-1)
    Channel 2: mean height Z     (clipped to [-3, 10] m, then normalised 0-1)

    Points are filtered to the BEV spatial extent BEFORE mapping,
    so old-code world-scale coordinates produce an empty (zero) grid.
    Only ego-centric points within ±32 m lateral and -16 to +48 m forward
    are used -- the range where rash events physically occur.
    """
    bev = np.zeros((3, BEV_H, BEV_W), dtype=np.float32)

    if not os.path.exists(npy_path):
        return bev
    pts = np.load(npy_path)
    if pts.ndim != 2 or pts.shape[1] < 4 or len(pts) == 0:
        return bev

    x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    # ── Spatial filter: keep only ego-centric points within BEV extent ──
    mask = ((x >= BEV_X_MIN) & (x < BEV_X_MAX) &
            (y >= BEV_Y_MIN) & (y < BEV_Y_MAX) &
            (z > -3.0) & (z < 10.0))
    if mask.sum() == 0:
        return bev   # world-scale data: returns blank BEV (model falls back to RGB/JSON)

    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    # ── Pixel coordinates ──
    col = np.clip(((x - BEV_X_MIN) / BEV_RES).astype(int), 0, BEV_W - 1)
    row = np.clip(((y - BEV_Y_MIN) / BEV_RES).astype(int), 0, BEV_H - 1)

    # ── Accumulate per-cell sums (vectorised) ──
    occ     = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    cnt     = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    sum_int = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    sum_z   = np.zeros((BEV_H, BEV_W), dtype=np.float32)

    np.add.at(occ,     (row, col), 1.0)
    np.add.at(cnt,     (row, col), 1.0)
    np.add.at(sum_int, (row, col), intensity)
    np.add.at(sum_z,   (row, col), z)

    valid = cnt > 0
    occ[valid]  = 1.0
    mean_int    = np.where(valid, sum_int / cnt, 0.0)
    mean_z_norm = np.where(valid,
                           np.clip((sum_z / cnt + 3.0) / 13.0, 0.0, 1.0), 0.0)

    bev[0] = occ
    bev[1] = mean_int
    bev[2] = mean_z_norm
    return bev


# ─────────────────────────────────────────────
# UNIFIED CACHE EXTRACTION
# ─────────────────────────────────────────────

def extract_and_cache(force: bool = False):
    """
    Phase 1a + 1b combined.
    Saves CACHE_ROOT/<ep>/<ev>/<FFFFF>.npz per frame with:
        rgb_feat  (576,)          frozen backbone, 6-view mean-pooled
        sem_feat  (576,)          same backbone on semantic images
        bev       (3, 64, 64)     float16 BEV grid from LiDAR
    """
    print("\n=== Phase 1: Feature extraction & BEV generation ===")
    backbone = build_backbone()
    clips    = sorted(glob.glob(f"{DATASET_ROOT}/episode_*/event_*"))
    total    = 0
    t0       = time.time()

    for clip_dir in clips:
        parts     = clip_dir.replace("\\", "/").split("/")
        ep_name   = parts[-2]
        ev_name   = parts[-1]
        cache_dir = os.path.join(CACHE_ROOT, ep_name, ev_name)
        os.makedirs(cache_dir, exist_ok=True)

        ann_files = sorted(glob.glob(os.path.join(clip_dir, "ann", "*.json")))
        for af in ann_files:
            frame_id   = os.path.splitext(os.path.basename(af))[0]
            cache_path = os.path.join(cache_dir, f"{frame_id}.npz")

            if os.path.exists(cache_path) and not force:
                total += 1
                continue

            try:
                rgb_feat, sem_feat = extract_both_modalities(backbone, clip_dir, frame_id)
                bev      = lidar_to_bev(
                    os.path.join(clip_dir, "lidar", f"{frame_id}.npy"))

                if rgb_feat is None:
                    rgb_feat = np.zeros(CAM_BB_DIM, dtype=np.float32)
                if sem_feat is None:
                    sem_feat = np.zeros(CAM_BB_DIM, dtype=np.float32)

                np.savez_compressed(cache_path,
                                    rgb_feat = rgb_feat,
                                    sem_feat = sem_feat,
                                    bev      = bev.astype(np.float16))
                total += 1
            except Exception as e:
                print(f"  WARN: skipped {frame_id} in {ep_name}/{ev_name}: {e}")

        elapsed = time.time() - t0
        fps     = total / max(elapsed, 1)
        print(f"  {ep_name}/{ev_name}  ({len(ann_files)} frames) "
              f"| total={total}  {fps:.1f} fr/s")

    print(f"Phase 1 done -- {total} frames cached  ({time.time()-t0:.0f}s)\n")


# ─────────────────────────────────────────────
# DATASET  (lazy-loading — no BEV in RAM)
# ─────────────────────────────────────────────

class FusionClip:
    """
    Per-clip data stored in RAM.

    BEV grids are kept as float16 (half precision) to halve memory:
      float16 BEV: 17 559 frames x 3 x 64 x 64 x 2 bytes  = 430 MB
      float32 BEV: same                                     = 860 MB
    Everything else is float32 but small (RGB+Sem: ~80 MB, JSON: ~1 MB).
    Total peak RAM: ~530 MB (vs ~2 GB with eager float32 + sliding windows).
    """
    __slots__ = ("episode", "rgb_feats", "sem_feats",
                 "bev_grids", "json_feats", "labels")

    def __init__(self, episode, rgb_feats, sem_feats,
                 bev_grids, json_feats, labels):
        self.episode    = episode
        self.rgb_feats  = rgb_feats     # (T, 576)          float32
        self.sem_feats  = sem_feats     # (T, 576)          float32
        self.bev_grids  = bev_grids     # (T, 3, 64, 64)   float16 -- half memory
        self.json_feats = json_feats    # (T, 19)           float32
        self.labels     = labels        # (T,)              int64


def load_all_clips() -> List[FusionClip]:
    """
    Load all clip data into RAM.
    BEV grids are stored as float16 to keep peak memory ~530 MB.
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

            rgb_list  = []; sem_list = []; bev_list = []
            json_list = []; label_list = []
            ok = True

            for af in ann_files:
                fid   = os.path.splitext(os.path.basename(af))[0]
                cpath = os.path.join(cache_dir, f"{fid}.npz")
                if not os.path.exists(cpath):
                    print(f"  WARN: cache missing {cpath}")
                    ok = False; break

                c = np.load(cpath)
                rgb_list.append(c["rgb_feat"].astype(np.float32))
                sem_list.append(c["sem_feat"].astype(np.float32))
                bev_list.append(c["bev"])            # keep as float16

                with open(af) as f:
                    ann = json.load(f)
                json_list.append(extract_json_features(ann))
                label_list.append(int(ann["anomaly"]["is_event_frame"]))

            if not ok or len(ann_files) < SEQ_LEN + 1:
                continue

            clips_out.append(FusionClip(
                episode    = ep_idx,
                rgb_feats  = np.array(rgb_list,   dtype=np.float32),
                sem_feats  = np.array(sem_list,   dtype=np.float32),
                bev_grids  = np.array(bev_list,   dtype=np.float16),  # half precision
                json_feats = np.array(json_list,  dtype=np.float32),
                labels     = np.array(label_list, dtype=np.int64),
            ))
            total_f += len(label_list)
            event_f += sum(label_list)

    n = total_f - event_f
    print(f"Loaded {len(clips_out)} clips | {total_f} frames "
          f"({event_f} event / {n} normal = {100*event_f/max(total_f,1):.1f}%)")
    return clips_out


class EarlyFusionDatasetV2(Dataset):
    """
    Sliding-window dataset.

    In-RAM BEV grids (float16) are sliced per window and converted to
    float32 in __getitem__ -- no disk I/O during training.
    """

    def __init__(self, clips: List[FusionClip]):
        self.samples: List[tuple] = []

        for clip in clips:
            T = len(clip.labels)
            for t in range(T):
                start = t - SEQ_LEN + 1
                pad   = max(-start, 0)
                idx   = max(start, 0)

                def _win(arr, p=pad, i=idx, tt=t):
                    slc = arr[i:tt+1]
                    if p > 0:
                        slc = np.concatenate(
                            [np.zeros((p,) + arr.shape[1:], dtype=arr.dtype), slc], 0)
                    return slc   # dtype matches source (float32 or float16)

                self.samples.append((
                    _win(clip.rgb_feats),    # (T, 576)         float32 view
                    _win(clip.sem_feats),    # (T, 576)         float32 view
                    _win(clip.bev_grids),    # (T, 3, 64, 64)   float16 view
                    _win(clip.json_feats),   # (T, 19)          float32 view
                    float(clip.labels[t]),
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        rgb_w, sem_w, bev_w16, json_w, label = self.samples[idx]
        # BEV: convert float16 -> float32 here (cheap, no disk I/O)
        bev_w = bev_w16.astype(np.float32)
        return (
            torch.from_numpy(rgb_w.copy()),
            torch.from_numpy(sem_w.copy()),
            torch.from_numpy(bev_w),
            torch.from_numpy(json_w.copy()),
            torch.tensor(label, dtype=torch.float32),
        )


def make_loaders(clips: List[FusionClip]):
    train = [c for c in clips if c.episode not in TEST_EPISODES | VAL_EPISODES]
    val   = [c for c in clips if c.episode in VAL_EPISODES]
    test  = [c for c in clips if c.episode in TEST_EPISODES]

    print(f"\nSplit -- train: {len(train)} clips | val: {len(val)} | test: {len(test)}")

    all_labels = np.concatenate([c.labels for c in train])
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    pos_w = float(n_neg) / max(float(n_pos), 1.0)
    print(f"pos_weight = {pos_w:.3f}  ({int(n_pos)} event / {int(n_neg)} normal in train)")

    kw = dict(num_workers=0, pin_memory=False)
    tr_dl = DataLoader(EarlyFusionDatasetV2(train), BATCH_SIZE, shuffle=True,  **kw)
    vl_dl = DataLoader(EarlyFusionDatasetV2(val),   BATCH_SIZE, shuffle=False, **kw)
    ts_dl = DataLoader(EarlyFusionDatasetV2(test),  BATCH_SIZE, shuffle=False, **kw)

    pw = torch.tensor([pos_w], dtype=torch.float32).to(DEVICE)
    return tr_dl, vl_dl, ts_dl, pw


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class BEVEncoder(nn.Module):
    """
    Trainable CNN: 3-channel BEV (3, 64, 64) -> 64-dim vector.

    Architecture:
      Conv(3->16, 3x3) + BN + ReLU + MaxPool2 -> (16, 32, 32)
      Conv(16->32, 3x3)+ BN + ReLU + MaxPool2 -> (32, 16, 16)
      Conv(32->64, 3x3)+ BN + ReLU            -> (64, 16, 16)
      AdaptiveAvgPool(4x4)                     -> (64, 4, 4)
      Dropout + Flatten + Linear(1024, 64) + ReLU

    ~89 K parameters. Dropout before the linear layer adds regularisation.
    """
    def __init__(self, out_dim: int = LIDAR_PROJ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(LIDAR_BEV_CH, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                             # (16, 32, 32)
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                             # (32, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                     # (64,  4,  4)
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),                         # regularise output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 64, 64)
        return self.net(x)


class EarlyFusionV2(nn.Module):
    """
    Four-modality early-fusion BiGRU.

    All four modalities are projected to compact vectors, concatenated
    at EVERY TIMESTEP, then processed jointly by a bidirectional GRU.
    This is "early fusion" because the cross-modal combination happens
    at the input to the temporal model, not at the output.

    Input shapes (per forward call):
        x_rgb   : (B, T, 576)   -- frozen backbone features
        x_sem   : (B, T, 576)   -- frozen backbone features (semantic)
        x_bev   : (B, T, 3, H, W) -- raw BEV grids (processed by BEVEncoder)
        x_json  : (B, T, 19)    -- JSON / kinematic feature vector

    Output: (B,) raw logit  (apply sigmoid for probability)
    """

    def __init__(self):
        super().__init__()

        # ── Modality encoders ──
        self.bev_enc = BEVEncoder(LIDAR_PROJ)          # trainable

        self.rgb_proj = nn.Sequential(
            nn.Linear(CAM_BB_DIM, RGB_PROJ_DIM),
            nn.LayerNorm(RGB_PROJ_DIM), nn.ReLU(),
        )
        self.sem_proj = nn.Sequential(
            nn.Linear(CAM_BB_DIM, SEM_PROJ_DIM),
            nn.LayerNorm(SEM_PROJ_DIM), nn.ReLU(),
        )
        self.json_proj = nn.Sequential(
            nn.Linear(JSON_DIM, JSON_PROJ),
            nn.LayerNorm(JSON_PROJ), nn.ReLU(),
        )

        # ── Temporal model ──
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
            nn.Linear(HIDDEN_DIM * 2, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x_rgb, x_sem, x_bev, x_json):
        B, T = x_rgb.shape[:2]

        # ── Project flat features (B*T, dim) -> (B, T, proj_dim) ──
        rgb_f  = self.rgb_proj (x_rgb .reshape(B*T, -1)).reshape(B, T, -1)
        sem_f  = self.sem_proj (x_sem .reshape(B*T, -1)).reshape(B, T, -1)
        json_f = self.json_proj(x_json.reshape(B*T, -1)).reshape(B, T, -1)

        # ── BEV CNN: (B, T, 3, H, W) -> (B, T, lidar_proj) ──
        bev_in  = x_bev.reshape(B*T, LIDAR_BEV_CH, BEV_H, BEV_W)
        bev_f   = self.bev_enc(bev_in).reshape(B, T, -1)

        # ── Early fusion: concat all four at every timestep (B, T, 192) ──
        fused = torch.cat([bev_f, rgb_f, sem_f, json_f], dim=-1)

        # ── Temporal reasoning ──
        out, _ = self.gru(fused)             # (B, T, 2*H)
        last   = out[:, -1, :]               # (B, 2*H) last timestep

        return self.head(last).squeeze(-1)   # (B,)


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, opt=None):
    model.train(opt is not None)
    total_loss = 0.0
    all_probs  = []; all_labels = []

    with torch.set_grad_enabled(opt is not None):
        for x_rgb, x_sem, x_bev, x_json, y in loader:
            x_rgb  = x_rgb.to(DEVICE)
            x_sem  = x_sem.to(DEVICE)
            x_bev  = x_bev.to(DEVICE)
            x_json = x_json.to(DEVICE)
            y      = y.to(DEVICE)

            logits = model(x_rgb, x_sem, x_bev, x_json)
            loss   = criterion(logits, y)

            if opt is not None:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item() * len(y)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(y.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return total_loss / max(len(loader.dataset), 1), probs, labels


def metrics(probs, labels, t=0.5):
    p = (probs >= t).astype(int)
    try:    auc = roc_auc_score(labels, probs)
    except: auc = float("nan")
    return {
        "f1":        f1_score       (labels, p, zero_division=0),
        "precision": precision_score(labels, p, zero_division=0),
        "recall":    recall_score   (labels, p, zero_division=0),
        "auc":       auc,
    }


def find_best_threshold(probs, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def save_plots(log, ts_labels, ts_probs, best_t):
    # Training curves
    ep = [e["epoch"] for e in log]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [e["train_loss"] for e in log], label="train")
    a1.plot(ep, [e["val_loss"]   for e in log], label="val")
    a1.set_title("Loss (Early Fusion v2)"); a1.legend()
    a2.plot(ep, [e["train_f1"] for e in log], label="train")
    a2.plot(ep, [e["val_f1"]   for e in log], label="val")
    a2.set_title("F1 (Early Fusion v2)"); a2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "ef2_training_curves.png"), dpi=120)
    plt.close(); print(f"  Saved: models/ef2_training_curves.png")

    # Confusion matrix
    cm = confusion_matrix(ts_labels, (ts_probs >= best_t).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Rash"], yticklabels=["Normal","Rash"])
    ax.set_title("Confusion Matrix -- Early Fusion v2 (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "ef2_confusion_matrix.png"), dpi=120)
    plt.close(); print(f"  Saved: models/ef2_confusion_matrix.png")

    # ROC
    try:
        fpr, tpr, _ = roc_curve(ts_labels, ts_probs)
        auc_val = roc_auc_score(ts_labels, ts_probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"Early Fusion v2  AUROC={auc_val:.3f}")
        ax.plot([0,1],[0,1],"k--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC (Early Fusion v2 -- Test)")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "ef2_roc_curve.png"), dpi=120)
        plt.close(); print(f"  Saved: models/ef2_roc_curve.png")
    except Exception as e:
        print(f"  ROC skipped: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(skip_cache: bool = False):
    if not skip_cache:
        extract_and_cache()
    else:
        print("Skipping cache extraction (--skip-cache).")

    # ── Load data ──
    clips = load_all_clips()
    if not clips:
        print("ERROR: No clips loaded. Run without --skip-cache first.")
        sys.exit(1)

    tr_dl, vl_dl, ts_dl, pw = make_loaders(clips)

    # ── Model ──
    model     = EarlyFusionV2().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt       = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n_total   = sum(p.numel() for p in model.parameters())
    n_train   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bev_par   = sum(p.numel() for p in model.bev_enc.parameters())
    print(f"\nModel : EarlyFusionV2")
    print(f"  Total params     : {n_total:,}")
    print(f"  Trainable params : {n_train:,}")
    print(f"  BEV CNN params   : {bev_par:,}  (trainable)")
    print(f"  Fused dim / frame: {FUSED_DIM} "
          f"(BEV={LIDAR_PROJ} + RGB={RGB_PROJ_DIM} + "
          f"Sem={SEM_PROJ_DIM} + JSON={JSON_PROJ})")
    print(f"  BiGRU            : {FUSED_DIM} -> hidden={HIDDEN_DIM} x2 "
          f"(bidirectional), {N_LAYERS} layers")
    print(f"Training max {EPOCHS} epochs, early-stop patience={PATIENCE}\n")

    best_val_f1  = -1.0
    no_improve   = 0           # epochs without val-F1 improvement (early stop)
    ckpt_path    = os.path.join(MODEL_DIR, "ef2_best.pt")
    log          = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_p, tr_l = run_epoch(model, tr_dl, criterion, opt)
        vl_loss, vl_p, vl_l = run_epoch(model, vl_dl, criterion)
        sched.step()

        tr_m = metrics(tr_p, tr_l)
        vl_m = metrics(vl_p, vl_l)
        log.append({"epoch": epoch,
                    "train_loss": round(tr_loss,5), "train_f1": round(tr_m["f1"],4),
                    "val_loss":   round(vl_loss,5), "val_f1":   round(vl_m["f1"],4),
                    "val_auc":    round(vl_m["auc"],4)})

        if vl_m["f1"] > best_val_f1:
            best_val_f1 = vl_m["f1"]
            no_improve  = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_f1": best_val_f1}, ckpt_path)
            star = " << best"
        else:
            no_improve += 1
            star = f"  (no improve {no_improve}/{PATIENCE})"

        if epoch % 5 == 0 or epoch == 1 or no_improve == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train: loss={tr_loss:.4f} F1={tr_m['f1']:.3f}  |  "
                  f"val: loss={vl_loss:.4f} F1={vl_m['f1']:.3f} "
                  f"AUC={vl_m['auc']:.3f}{star}")

        if no_improve >= PATIENCE:
            print(f"Early stop at epoch {epoch} (no val-F1 gain for {PATIENCE} epochs).")
            break

    print(f"\nBest val F1: {best_val_f1:.4f}  (checkpoint: {ckpt_path})")

    with open(os.path.join(MODEL_DIR, "ef2_training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # ── Test evaluation ──
    print("\n--- Test set evaluation ---")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, vl_p, vl_l = run_epoch(model, vl_dl, criterion)
    _, ts_p, ts_l = run_epoch(model, ts_dl, criterion)

    best_t, _ = find_best_threshold(vl_p, vl_l)
    print(f"Optimal threshold (val set): {best_t:.2f}")

    if ts_l.sum() == 0:
        print("WARNING: No event frames in test set. Adjust TEST_EPISODES.")
    else:
        ts_m = metrics(ts_p, ts_l, t=best_t)
        print(f"\n  F1        : {ts_m['f1']:.4f}")
        print(f"  Precision : {ts_m['precision']:.4f}")
        print(f"  Recall    : {ts_m['recall']:.4f}")
        print(f"  AUROC     : {ts_m['auc']:.4f}")
        print()
        print(classification_report(ts_l, (ts_p >= best_t).astype(int),
                                    target_names=["Normal","Rash Event"],
                                    zero_division=0))
        save_plots(log, ts_l, ts_p, best_t)

    # ── Comparison table ──
    print("\n--- Model comparison (val F1) ---")
    for fname, label in [("kinematic_gru_best.pt",  "Kinematics only       "),
                         ("early_fusion_best.pt",   "Early Fusion v1 (RGB+Kin+LiDAR)"),
                         ("ef2_best.pt",            "Early Fusion v2 (BEV+RGB+Sem+JSON)")]:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            c = torch.load(p, map_location="cpu", weights_only=False)
            print(f"  {label}: val F1 = {c['val_f1']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    main(skip_cache=args.skip_cache)
