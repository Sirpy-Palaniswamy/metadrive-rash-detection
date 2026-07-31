"""
Generates dataset_exploration.ipynb
Run:  python build_notebook.py
Then open dataset_exploration.ipynb in Jupyter / VS Code.
"""

import nbformat as nbf
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "dataset_exploration.ipynb")

nb    = nbf.v4.new_notebook()
cells = []

def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ==============================================================================
md("""\
# MetaDrive Rash Driving Dataset - Visual Technical Exploration

**Project:** Simulation-Based Rash Driving Behaviour Prediction
**Modalities:** BEV LiDAR | Surround RGB (6 views) | Surround Semantic (6 views) | JSON Kinematics
**Simulator:** MetaDrive 0.4.3 | Python 3.8 | PyTorch 2.4.1

---
Sections:
1. Dataset overview and statistics
2. Class balance
3. Train / Val / Test split
4. BEV LiDAR grid (3 channels)
5. RGB camera surround views
6. Semantic segmentation views
7. LiDAR raw point cloud and projection
8. Kinematic feature distributions
9. Temporal patterns around rash events
10. Feature cache statistics
11. Model comparison results
12. Detailed EF-v2 metrics
13. Pre-saved training artefacts
14. Summary table
15. Gallery - all saved plots
""")

# ------------------------------------------------------------------------------
md("## 0 - Imports and Paths")
code("""\
%matplotlib inline

import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
from IPython.display import display, Image as IPImage

# Colour palette
C_BLUE   = "#0f3460"
C_MID    = "#16213e"
C_ACCENT = "#e94560"
C_LIGHT  = "#a8d8ea"
C_GREEN  = "#2ecc71"
C_ORANGE = "#e67e22"

sns.set_theme(style="whitegrid", palette="deep")

# Resolve notebook directory (works both as script and inside Jupyter)
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()

DATASET_ROOT = os.path.join(BASE, "metadrive_fusion_dataset")
CACHE_ROOT   = os.path.join(BASE, "feature_cache_v2")
MODEL_DIR    = os.path.join(BASE, "models")

TEST_EPS = {9, 10, 47, 49}
VAL_EPS  = {3, 6, 45, 46}

print("BASE        :", BASE)
print("Dataset root:", DATASET_ROOT)
print("Exists      :", os.path.exists(DATASET_ROOT))
""")

# ------------------------------------------------------------------------------
md("## 1 - Scan All Annotations")
code("""\
records = []

for ep_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, "episode_*"))):
    ep_idx = int(os.path.basename(ep_dir).split("_")[1])
    split  = ("test" if ep_idx in TEST_EPS
              else "val" if ep_idx in VAL_EPS else "train")

    for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "event_*"))):
        ann_files = sorted(glob.glob(os.path.join(ev_dir, "ann", "*.json")))
        for af in ann_files:
            with open(af) as f:
                ann = json.load(f)

            ego_speed = ann["ego"]["speed_ms"]
            is_event  = int(ann["anomaly"]["is_event_frame"])

            npcs = sorted(ann.get("objects", []),
                          key=lambda o: o.get("distance_to_ego", 9999))
            npc1 = npcs[0] if npcs else None

            records.append(dict(
                episode    = ep_idx,
                event_clip = os.path.basename(ev_dir),
                split      = split,
                is_event   = is_event,
                ego_speed  = ego_speed,
                n_npcs     = len(npcs),
                npc1_dist  = npc1["distance_to_ego"] if npc1 else np.nan,
                npc1_speed = npc1["speed_ms"]         if npc1 else np.nan,
                is_rogue   = any(o.get("is_rogue", False) for o in npcs),
            ))

import pandas as pd
df = pd.DataFrame(records)

print(f"Total frames  : {len(df):,}")
print(f"Event frames  : {df.is_event.sum():,}  ({df.is_event.mean()*100:.1f}%)")
print(f"Normal frames : {(~df.is_event.astype(bool)).sum():,}")
print(f"Episodes      : {df.episode.nunique()}")
print(f"Event clips   : {df.groupby(['episode','event_clip']).ngroups}")
df.head(3)
""")

# ------------------------------------------------------------------------------
md("""\
## 2 - Dataset Overview Statistics

Four panels:
- (a) Total frames per episode, coloured by split
- (b) Event clip count per episode
- (c) Overall class balance (Normal vs Rash Event)
- (d) Train / Val / Test frame proportion (pie)
""")
code("""\
ep_stats = df.groupby("episode").agg(
    n_frames = ("is_event", "count"),
    n_event  = ("is_event", "sum"),
    n_clips  = ("event_clip", "nunique"),
    split    = ("split", "first"),
).reset_index()

split_colors = {"train": C_BLUE, "val": C_ORANGE, "test": C_ACCENT}
bar_colors   = [split_colors[s] for s in ep_stats["split"]]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold", color=C_MID, y=1.01)

# (a) frames per episode
ax = axes[0, 0]
ax.bar(ep_stats.episode, ep_stats.n_frames, color=bar_colors,
       edgecolor="white", linewidth=0.4)
ax.set_xlabel("Episode index")
ax.set_ylabel("Total frames")
ax.set_title("(a) Frames per Episode  [colour = split]")
legend_patches = [mpatches.Patch(color=v, label=k.capitalize())
                  for k, v in split_colors.items()]
ax.legend(handles=legend_patches, fontsize=9)
ax.set_xlim(-1, ep_stats.episode.max() + 1)

# (b) clips per episode
ax = axes[0, 1]
ax.bar(ep_stats.episode, ep_stats.n_clips, color=bar_colors,
       edgecolor="white", linewidth=0.4)
ax.set_xlabel("Episode index")
ax.set_ylabel("Number of event clips")
ax.set_title("(b) Event Clips per Episode")

# (c) class balance
ax = axes[1, 0]
class_counts = df.is_event.value_counts().sort_index()
bars = ax.bar(["Normal (0)", "Rash Event (1)"],
              class_counts.values,
              color=[C_LIGHT, C_ACCENT], edgecolor=C_MID, linewidth=0.8, width=0.5)
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
            f"{val:,}\\n({val / len(df) * 100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Frame count")
ax.set_title("(c) Class Balance")
ax.set_ylim(0, class_counts.max() * 1.15)

# (d) split pie
ax = axes[1, 1]
split_counts = (df.groupby("split")["is_event"]
                  .count()
                  .reindex(["train", "val", "test"]))
wedge_colors = [split_colors[s] for s in split_counts.index]
wedges, texts, autotexts = ax.pie(
    split_counts.values,
    labels=split_counts.index.str.capitalize(),
    colors=wedge_colors, autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 11})
for at in autotexts:
    at.set_fontsize(10)
    at.set_color("white")
    at.set_fontweight("bold")
ax.set_title("(d) Train / Val / Test Split (frames)")

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "nb_overview.png"), dpi=130, bbox_inches="tight")
plt.show()
print("Saved: models/nb_overview.png")
""")

# ------------------------------------------------------------------------------
md("## 3 - Class Balance Per Split")
code("""\
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
fig.suptitle("Event vs Normal frame counts by split",
             fontsize=13, fontweight="bold", color=C_MID)

for ax, split in zip(axes, ["train", "val", "test"]):
    sub    = df[df.split == split]
    counts = sub.is_event.value_counts().sort_index()
    bars   = ax.bar(["Normal", "Rash Event"],
                    counts.values,
                    color=[C_LIGHT, C_ACCENT], edgecolor=C_MID,
                    linewidth=0.7, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}\\n({val / len(sub) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_title(f"{split.capitalize()}  (n={len(sub):,})",
                 fontweight="bold", color=split_colors[split])
    ax.set_ylabel("Frames")
    ax.set_ylim(0, counts.max() * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "nb_class_balance.png"), dpi=130, bbox_inches="tight")
plt.show()
""")

# ------------------------------------------------------------------------------
md("""\
## 4 - BEV LiDAR Grid Visualisation

Each frame's LiDAR point cloud is projected onto a 64x64 pixel top-down grid
with 3 channels:

- **Channel 0 - Occupancy:** 1 if any LiDAR point fell in this cell
- **Channel 1 - Mean Intensity:** average reflectance (0 to 1)
- **Channel 2 - Mean Height:** normalised Z value

Ego vehicle sits at pixel (32, 16) -- lower-centre, forward-biased coverage.

Grid spec: X in [-32, +32] m (lateral) | Y in [-16, +48] m (forward)
""")
code("""\
# Find an event clip from ep >= 19 (valid ego-centric LiDAR)
sample_npz = None
for ep_dir in sorted(glob.glob(os.path.join(CACHE_ROOT, "episode_*"))):
    ep_idx = int(os.path.basename(ep_dir).split("_")[1])
    if ep_idx < 19:
        continue
    npz_files = sorted(glob.glob(os.path.join(ep_dir, "*", "*.npz")))
    if npz_files:
        sample_npz = npz_files[len(npz_files) // 2]
        break

if sample_npz is None:
    print("No valid cache file found (ep >= 19)")
else:
    data = np.load(sample_npz)
    bev  = data["bev"].astype(np.float32)   # float16 -> float32
    print(f"File  : {sample_npz}")
    print(f"Shape : {bev.shape}  dtype: {bev.dtype}")
    print(f"Occupancy filled : {(bev[0] > 0).sum()} / {64 * 64} pixels")
    print(f"Intensity range  : [{bev[1].min():.3f}, {bev[1].max():.3f}]")
    print(f"Height range     : [{bev[2].min():.3f}, {bev[2].max():.3f}]")

    ch_names = ["Occupancy", "Mean Intensity", "Mean Height"]
    ch_cmaps = ["Blues", "plasma", "viridis"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("BEV LiDAR Grid  |  64x64 px, 1 m/px  |  ego at pixel (32, 16)",
                 fontsize=12, fontweight="bold", color=C_MID)

    for i, (ax, name, cmap) in enumerate(zip(axes[:3], ch_names, ch_cmaps)):
        im = ax.imshow(bev[i], cmap=cmap, origin="lower", aspect="equal")
        ax.scatter([32], [16], marker="*", s=120, c=C_ACCENT,
                   zorder=5, label="Ego vehicle")
        ax.set_title(f"Ch {i}: {name}", fontweight="bold")
        ax.set_xlabel("X pixel (lateral)")
        ax.set_ylabel("Y pixel (forward)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if i == 0:
            ax.legend(fontsize=8)

    # Composite: R=occupancy, G=intensity, B=height
    ax = axes[3]
    h_max = bev[2].max()
    composite = np.stack([
        bev[0],
        np.clip(bev[1], 0, 1),
        np.clip(bev[2] / (h_max + 1e-6), 0, 1),
    ], axis=-1)
    ax.imshow(composite, origin="lower", aspect="equal")
    ax.scatter([32], [16], marker="*", s=120, c="yellow", zorder=5, label="Ego")
    ax.set_title("Composite (R=Occ, G=Int, B=Ht)", fontweight="bold")
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "nb_bev_channels.png"), dpi=130, bbox_inches="tight")
    plt.show()
""")

code("""\
# Compare: Normal frame vs Rash Event frame (same clip)
event_npz    = None
nonevent_npz = None

for ep_dir in sorted(glob.glob(os.path.join(CACHE_ROOT, "episode_*"))):
    ep_idx = int(os.path.basename(ep_dir).split("_")[1])
    if ep_idx < 19:
        continue
    for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "*"))):
        ep_name = os.path.basename(ep_dir)
        ev_name = os.path.basename(ev_dir)
        ann_dir = os.path.join(DATASET_ROOT, ep_name, ev_name, "ann")
        if not os.path.exists(ann_dir):
            continue
        anns = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
        npzs = sorted(glob.glob(os.path.join(ev_dir, "*.npz")))
        if len(anns) != len(npzs) or not anns:
            continue
        for a, n in zip(anns, npzs):
            with open(a) as f:
                is_ev = json.load(f)["anomaly"]["is_event_frame"]
            if is_ev and event_npz is None:
                event_npz = n
            if not is_ev and nonevent_npz is None:
                nonevent_npz = n
        if event_npz and nonevent_npz:
            break
    if event_npz and nonevent_npz:
        break

if event_npz and nonevent_npz:
    bev_ev = np.load(event_npz)["bev"].astype(np.float32)
    bev_ne = np.load(nonevent_npz)["bev"].astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("BEV Comparison: Normal Frame (top) vs Rash Event Frame (bottom)",
                 fontsize=12, fontweight="bold", color=C_MID)

    titles    = ([f"Normal -- Ch {i}"     for i in range(3)] +
                 [f"Rash Event -- Ch {i}" for i in range(3)])
    data_list = [bev_ne[0], bev_ne[1], bev_ne[2],
                 bev_ev[0], bev_ev[1], bev_ev[2]]
    cmaps     = ["Blues", "plasma", "viridis"] * 2

    for ax, title, dat, cmap in zip(axes.ravel(), titles, data_list, cmaps):
        im = ax.imshow(dat, cmap=cmap, origin="lower", aspect="equal")
        ax.scatter([32], [16], marker="*", s=90, c=C_ACCENT, zorder=5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("X px")
        ax.set_ylabel("Y px")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "nb_bev_comparison.png"), dpi=130, bbox_inches="tight")
    plt.show()
else:
    print("Could not find matching event / non-event pair -- check dataset paths")
""")

# ------------------------------------------------------------------------------
md("""\
## 5 - Surround RGB Camera Views

6 cameras provide a full 360 degree surround view.
MobileNetV3-Small processes each view; the 6 feature vectors are mean-pooled
into a single 576-dim representation per frame.

Layout: front-left | front | front-right (top row)
        back-left  | back  | back-right  (bottom row)
""")
code("""\
CAMS = ["front", "front_left", "front_right", "back", "back_left", "back_right"]
CAM_LAYOUT = [(0, 1), (0, 0), (0, 2), (1, 1), (1, 0), (1, 2)]

cam_labels = {
    "front":       "FRONT",
    "front_left":  "FRONT LEFT",
    "front_right": "FRONT RIGHT",
    "back":        "BACK",
    "back_left":   "BACK LEFT",
    "back_right":  "BACK RIGHT",
}

sample_rgb_dir = None
sample_fid     = None

for ep_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, "episode_*"))):
    ep_idx = int(os.path.basename(ep_dir).split("_")[1])
    if ep_idx < 19:
        continue
    for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "event_*"))):
        anns = sorted(glob.glob(os.path.join(ev_dir, "ann", "*.json")))
        for af in anns:
            with open(af) as f:
                ann = json.load(f)
            if ann["anomaly"]["is_event_frame"]:
                fid  = os.path.splitext(os.path.basename(af))[0]
                imgs = [os.path.join(ev_dir, "rgb", f"{fid}_{c}.png") for c in CAMS]
                if all(os.path.exists(p) for p in imgs):
                    sample_rgb_dir = os.path.join(ev_dir, "rgb")
                    sample_fid     = fid
                    break
        if sample_fid:
            break
    if sample_fid:
        break

if sample_fid:
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"Surround RGB Views -- Rash Event Frame  (fid={sample_fid})\\n"
                  "Front camera at top-centre; back camera at bottom-centre",
                 fontsize=12, fontweight="bold", color=C_MID)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.05)

    for cam, (row, col) in zip(CAMS, CAM_LAYOUT):
        ax   = fig.add_subplot(gs[row, col])
        path = os.path.join(sample_rgb_dir, f"{sample_fid}_{cam}.png")
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(cam_labels[cam], fontweight="bold", color=C_BLUE, fontsize=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(C_ACCENT)
            spine.set_linewidth(2.5)

    plt.savefig(os.path.join(MODEL_DIR, "nb_rgb_views.png"), dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Event frame displayed: {sample_fid}")
else:
    print("No matching event frame found with all 6 RGB cameras")
""")

# ------------------------------------------------------------------------------
md("""\
## 6 - Surround Semantic Segmentation Views

Same 6-view layout using semantic segmentation images.
Each colour maps to an object class (road, vehicle, sky, building, ...).
A **separate** MobileNetV3-Small head processes these views,
producing a 32-dim semantic feature after projection.
""")
code("""\
if sample_fid:
    # Build semantic dir from rgb dir using os.path (Windows-safe)
    ev_root = os.path.dirname(sample_rgb_dir)   # event_XXXX folder
    sem_dir = os.path.join(ev_root, "semantic")

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"Surround Semantic Segmentation Views -- same frame (fid={sample_fid})",
                 fontsize=12, fontweight="bold", color=C_MID)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.05)

    for cam, (row, col) in zip(CAMS, CAM_LAYOUT):
        ax   = fig.add_subplot(gs[row, col])
        path = os.path.join(sem_dir, f"{sample_fid}_{cam}.png")
        if os.path.exists(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(cam_labels[cam], fontweight="bold", color=C_MID, fontsize=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GREEN)
            spine.set_linewidth(2.5)

    plt.savefig(os.path.join(MODEL_DIR, "nb_sem_views.png"), dpi=130, bbox_inches="tight")
    plt.show()
else:
    print("Run Section 5 first to set sample_fid")
""")

# ------------------------------------------------------------------------------
md("""\
## 7 - Raw LiDAR Point Cloud and BEV Projection

MetaDrive renders 6 depth images at 60 degree intervals and back-projects them
into (X, Y, Z, intensity) 3D points in ego-centric coordinates.

- Left plot: top-down XY view with the BEV grid extent shown as a dashed box
- Centre: side YZ view (height profile)
- Right: intensity value histogram
""")
code("""\
lidar_path = None
for ep_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, "episode_*"))):
    ep_idx = int(os.path.basename(ep_dir).split("_")[1])
    if ep_idx < 19:
        continue
    files = sorted(glob.glob(os.path.join(ep_dir, "*", "lidar", "*.npy")))
    if files:
        lidar_path = files[len(files) // 2]
        break

if lidar_path:
    pts = np.load(lidar_path)   # (N, 4): X, Y, Z, intensity
    print(f"File  : {lidar_path}")
    print(f"Points: {pts.shape[0]:,}  columns: X, Y, Z, intensity")
    print(f"X in [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}] m")
    print(f"Y in [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}] m")
    print(f"Z in [{pts[:,2].min():.1f}, {pts[:,2].max():.1f}] m")

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("Raw LiDAR Point Cloud (ego-centric)  |  ego = origin (0, 0)",
                 fontsize=12, fontweight="bold", color=C_MID)

    # Top-down XY
    ax1 = fig.add_subplot(1, 3, 1)
    sc  = ax1.scatter(pts[:, 0], pts[:, 1], c=pts[:, 3], cmap="plasma",
                      s=0.8, alpha=0.6, rasterized=True)
    ax1.scatter([0], [0], marker="*", c=C_ACCENT, s=200, zorder=5, label="Ego")
    rect = plt.Rectangle((-32, -16), 64, 64, linewidth=1.5,
                         edgecolor=C_BLUE, facecolor="none",
                         linestyle="--", label="BEV extent")
    ax1.add_patch(rect)
    ax1.set_xlabel("X (m) lateral")
    ax1.set_ylabel("Y (m) forward")
    ax1.set_title("Top-Down (XY)  colour=intensity")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")
    plt.colorbar(sc, ax=ax1, fraction=0.046, label="Intensity")

    # Side view YZ
    ax2 = fig.add_subplot(1, 3, 2)
    sc2 = ax2.scatter(pts[:, 1], pts[:, 2], c=pts[:, 3], cmap="plasma",
                      s=0.8, alpha=0.6, rasterized=True)
    ax2.scatter([0], [0], marker="*", c=C_ACCENT, s=200, zorder=5)
    ax2.set_xlabel("Y (m) forward")
    ax2.set_ylabel("Z (m) height")
    ax2.set_title("Side View (YZ)  colour=intensity")
    plt.colorbar(sc2, ax=ax2, fraction=0.046, label="Intensity")

    # Intensity histogram
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(pts[:, 3], bins=60, color=C_BLUE, alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax3.set_xlabel("Intensity")
    ax3.set_ylabel("Point count")
    ax3.set_title(f"Intensity Distribution  ({pts.shape[0]:,} points)")
    ax3.axvline(pts[:, 3].mean(), color=C_ACCENT, lw=1.5,
                label=f"Mean = {pts[:,3].mean():.3f}")
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "nb_lidar_raw.png"), dpi=130, bbox_inches="tight")
    plt.show()
else:
    print("No LiDAR .npy file found for ep >= 19")
""")

# ------------------------------------------------------------------------------
md("""\
## 8 - Kinematic Feature Distributions

The 19-dim JSON kinematic vector covers:
- Ego: speed (m/s), heading (rad), position x/y
- NPC 1/2/3: relative x/y, speed, heading, distance

These are the sole inputs for the Kinematic LSTM baseline and the JSON branch
of Early Fusion V2.
""")
code("""\
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Kinematic Feature Distributions (red = Rash Event, blue = Normal)",
             fontsize=13, fontweight="bold", color=C_MID)

ev_mask = df["is_event"].astype(bool)
kw      = dict(alpha=0.55, bins=50, density=True)

# (a) Ego speed
ax = axes[0, 0]
ax.hist(df.loc[~ev_mask, "ego_speed"], color=C_LIGHT,  label="Normal",     **kw)
ax.hist(df.loc[ ev_mask, "ego_speed"], color=C_ACCENT, label="Rash Event", **kw)
ax.set_xlabel("Ego speed (m/s)")
ax.set_ylabel("Density")
ax.set_title("(a) Ego Speed")
ax.legend(fontsize=9)

# (b) Nearest NPC distance
ax = axes[0, 1]
ax.hist(df.loc[~ev_mask, "npc1_dist"].dropna(), color=C_LIGHT,  label="Normal",     **kw)
ax.hist(df.loc[ ev_mask, "npc1_dist"].dropna(), color=C_ACCENT, label="Rash Event", **kw)
ax.set_xlabel("Nearest NPC distance (m)")
ax.set_ylabel("Density")
ax.set_title("(b) Nearest NPC Distance")
ax.legend(fontsize=9)

# (c) Nearest NPC speed
ax = axes[0, 2]
ax.hist(df.loc[~ev_mask, "npc1_speed"].dropna(), color=C_LIGHT,  label="Normal",     **kw)
ax.hist(df.loc[ ev_mask, "npc1_speed"].dropna(), color=C_ACCENT, label="Rash Event", **kw)
ax.set_xlabel("Nearest NPC speed (m/s)")
ax.set_ylabel("Density")
ax.set_title("(c) Nearest NPC Speed")
ax.legend(fontsize=9)

# (d) NPC count per frame
ax = axes[1, 0]
bins_n = range(0, int(df.n_npcs.max()) + 2)
ax.hist(df.loc[~ev_mask, "n_npcs"], bins=bins_n, color=C_LIGHT,
        alpha=0.6, density=True, label="Normal")
ax.hist(df.loc[ ev_mask, "n_npcs"], bins=bins_n, color=C_ACCENT,
        alpha=0.6, density=True, label="Rash Event")
ax.set_xlabel("NPC count visible")
ax.set_ylabel("Density")
ax.set_title("(d) NPC Count per Frame")
ax.legend(fontsize=9)

# (e) Ego speed boxplot by split
ax = axes[1, 1]
df.boxplot(column="ego_speed", by="split",
           positions=[0, 1, 2], widths=0.45, ax=ax,
           patch_artist=True,
           boxprops=dict(facecolor=C_LIGHT, color=C_MID),
           medianprops=dict(color=C_ACCENT, linewidth=2))
plt.sca(ax)
plt.title("(e) Ego Speed by Split")
ax.set_xlabel("")
ax.set_ylabel("m/s")

# (f) Ego speed vs NPC distance scatter
ax = axes[1, 2]
samp   = df.dropna(subset=["npc1_dist"]).sample(min(3000, len(df)), random_state=42)
colors = [C_ACCENT if e else C_LIGHT for e in samp.is_event]
ax.scatter(samp.ego_speed, samp.npc1_dist, c=colors, s=4, alpha=0.5, rasterized=True)
ax.set_xlabel("Ego speed (m/s)")
ax.set_ylabel("Nearest NPC distance (m)")
ax.set_title("(f) Ego Speed vs NPC Distance\\n(red=event, blue=normal)")

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "nb_kinematics.png"), dpi=130, bbox_inches="tight")
plt.show()
""")

# ------------------------------------------------------------------------------
md("""\
## 9 - Temporal Patterns Around Rash Events

All event clips are aligned so t=0 is the first event frame.
The shaded band shows +/- one standard deviation across clips.
The red dashed line marks t=0; the red shaded region is the event window.
""")
code("""\
WINDOW_BEFORE = 30
WINDOW_AFTER  = 20

clip_series = []

for ep_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, "episode_*"))):
    for ev_dir in sorted(glob.glob(os.path.join(ep_dir, "event_*"))):
        anns = sorted(glob.glob(os.path.join(ev_dir, "ann", "*.json")))
        if not anns:
            continue
        frames = []
        for af in anns:
            with open(af) as f:
                ann = json.load(f)
            npcs = sorted(ann.get("objects", []),
                          key=lambda o: o.get("distance_to_ego", 9999))
            npc1 = npcs[0] if npcs else None
            frames.append(dict(
                is_event  = int(ann["anomaly"]["is_event_frame"]),
                ego_speed = ann["ego"]["speed_ms"],
                npc1_dist = npc1["distance_to_ego"] if npc1 else np.nan,
                npc1_spd  = npc1["speed_ms"]         if npc1 else np.nan,
            ))
        labels   = [fr["is_event"] for fr in frames]
        ev_start = next((i for i, l in enumerate(labels) if l == 1), None)
        if ev_start is None:
            continue
        for t, fr in enumerate(frames):
            clip_series.append(dict(rel_t=t - ev_start, **fr))

ts     = pd.DataFrame(clip_series)
ts_grp = ts.groupby("rel_t").agg(
    ego_speed_mean = ("ego_speed", "mean"),
    ego_speed_std  = ("ego_speed", "std"),
    npc1_dist_mean = ("npc1_dist", "mean"),
    npc1_dist_std  = ("npc1_dist", "std"),
    npc1_spd_mean  = ("npc1_spd",  "mean"),
    npc1_spd_std   = ("npc1_spd",  "std"),
    count          = ("is_event",  "count"),
).reset_index()

ts_plot = ts_grp[
    (ts_grp.rel_t >= -WINDOW_BEFORE) &
    (ts_grp.rel_t <= WINDOW_AFTER)   &
    (ts_grp["count"] >= 5)
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Temporal Patterns Around Rash Events  (t=0 = first event frame)",
             fontsize=13, fontweight="bold", color=C_MID)

def plot_ts(ax, y_mean, y_std, ylabel, title):
    t = ts_plot.rel_t.values
    m = ts_plot[y_mean].values
    s = ts_plot[y_std].fillna(0).values
    ax.fill_between(t, m - s, m + s, alpha=0.20, color=C_BLUE)
    ax.plot(t, m, color=C_BLUE, linewidth=2)
    ax.axvline(0, color=C_ACCENT, lw=1.8, linestyle="--", label="Event start (t=0)")
    ax.axvspan(0, WINDOW_AFTER, alpha=0.07, color=C_ACCENT)
    ax.set_xlabel("Frame relative to event start")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)

plot_ts(axes[0], "ego_speed_mean", "ego_speed_std",  "Ego speed (m/s)", "Ego Speed over Time")
plot_ts(axes[1], "npc1_dist_mean", "npc1_dist_std",  "Distance (m)",    "Nearest NPC Distance")
plot_ts(axes[2], "npc1_spd_mean",  "npc1_spd_std",   "NPC speed (m/s)", "Nearest NPC Speed")

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "nb_temporal.png"), dpi=130, bbox_inches="tight")
plt.show()
print(f"Max clips at any time step: {ts_grp['count'].max()}")
""")

# ------------------------------------------------------------------------------
md("""\
## 10 - Feature Cache Statistics

`feature_cache_v2/` holds pre-extracted features for all 17,559 frames.
Each `.npz` file contains three arrays:

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `rgb_feat` | (576,) | float32 | Mean-pooled MobileNetV3-Small RGB features |
| `sem_feat` | (576,) | float32 | Mean-pooled MobileNetV3-Small semantic features |
| `bev` | (3, 64, 64) | float16 | BEV LiDAR grid (half-precision) |

Every 25th file is sampled here for speed.
""")
code("""\
sample_npzs = []
for ep_dir in sorted(glob.glob(os.path.join(CACHE_ROOT, "episode_*"))):
    npzs = sorted(glob.glob(os.path.join(ep_dir, "*", "*.npz")))
    sample_npzs.extend(npzs[::25])

print(f"Sampling {len(sample_npzs)} files ...")

rgb_norms = []
sem_norms = []
bev_occ   = []
for path in sample_npzs:
    d = np.load(path)
    rgb_norms.append(np.linalg.norm(d["rgb_feat"]))
    sem_norms.append(np.linalg.norm(d["sem_feat"]))
    bev_occ.append(float((d["bev"][0] > 0).mean()))

rgb_norms = np.array(rgb_norms)
sem_norms = np.array(sem_norms)
bev_occ   = np.array(bev_occ)

print(f"RGB feat norm  -- mean: {rgb_norms.mean():.2f}  std: {rgb_norms.std():.2f}")
print(f"Sem feat norm  -- mean: {sem_norms.mean():.2f}  std: {sem_norms.std():.2f}")
print(f"BEV occupancy  -- mean: {bev_occ.mean():.3f}  (fraction of 64x64 cells with points)")
print(f"BEV empty (0%) -- {(bev_occ < 0.001).mean()*100:.1f}% of sampled frames")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Feature Cache Statistics (sampled every 25th file)",
             fontsize=12, fontweight="bold", color=C_MID)

axes[0].hist(rgb_norms, bins=40, color=C_BLUE, alpha=0.8, edgecolor="white")
axes[0].set_xlabel("L2 norm")
axes[0].set_ylabel("Count")
axes[0].set_title("RGB Feature Vector Norms (576-dim)")

axes[1].hist(sem_norms, bins=40, color=C_GREEN, alpha=0.8, edgecolor="white")
axes[1].set_xlabel("L2 norm")
axes[1].set_ylabel("Count")
axes[1].set_title("Semantic Feature Vector Norms (576-dim)")

axes[2].hist(bev_occ * 100, bins=40, color=C_ORANGE, alpha=0.8, edgecolor="white")
axes[2].set_xlabel("BEV cells occupied (%)")
axes[2].set_ylabel("Count")
axes[2].set_title("BEV Grid Occupancy\\n(0% = empty / old-episode LiDAR)")
axes[2].axvline(bev_occ.mean() * 100, color=C_ACCENT, lw=1.5,
                label=f"Mean = {bev_occ.mean()*100:.1f}%")
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "nb_cache_stats.png"), dpi=130, bbox_inches="tight")
plt.show()
""")

# ------------------------------------------------------------------------------
md("""\
## 11 - Model Comparison

Three model generations, increasing in complexity.
Val F1 scores are read from the saved `.pt` checkpoints.
Test F1 / AUROC for EF-v2 come from `eval_ef2.py`.
""")
code("""\
import torch

models_info = [
    dict(name="Kinematic\\nLSTM",
         fname="kinematic_gru_best.pt",
         params=184_000, val_f1=0.9275, test_f1=0.9450, test_auroc=0.9088,
         modalities="JSON (19-dim)"),
    dict(name="Early\\nFusion v1",
         fname="early_fusion_best.pt",
         params=184_961, val_f1=0.9265, test_f1=0.7657, test_auroc=0.8927,
         modalities="RGB + JSON + LiDAR stats"),
    dict(name="Early\\nFusion v2\\n(Ours)",
         fname="ef2_best.pt",
         params=492_193, val_f1=0.9455, test_f1=0.9003, test_auroc=0.8976,
         modalities="BEV + RGB + Sem + JSON"),
]

for m in models_info:
    p = os.path.join(MODEL_DIR, m["fname"])
    m["epoch"] = (torch.load(p, map_location="cpu", weights_only=False).get("epoch", "?")
                  if os.path.exists(p) else "?")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle("Model Comparison -- All Three Generations",
             fontsize=14, fontweight="bold", color=C_MID, y=1.02)

names   = [m["name"]        for m in models_info]
val_f1  = [m["val_f1"]      for m in models_info]
test_f1 = [m["test_f1"]     for m in models_info]
auroc   = [m["test_auroc"]  for m in models_info]
colors  = [C_LIGHT, C_ORANGE, C_ACCENT]
bar_kw  = dict(edgecolor=C_MID, linewidth=0.8, width=0.5)

def annotate_bars(ax, bars, values):
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_MID)

axes[0].bar(names, val_f1,  color=colors, **bar_kw)
annotate_bars(axes[0], axes[0].patches, val_f1)
axes[0].set_ylim(0.88, 0.97)
axes[0].set_ylabel("Val F1")
axes[0].set_title("Validation F1", fontweight="bold")

axes[1].bar(names, test_f1, color=colors, **bar_kw)
annotate_bars(axes[1], axes[1].patches, test_f1)
axes[1].set_ylim(0.70, 0.97)
axes[1].set_ylabel("Test F1")
axes[1].set_title("Test Set F1", fontweight="bold")

axes[2].bar(names, auroc,   color=colors, **bar_kw)
annotate_bars(axes[2], axes[2].patches, auroc)
axes[2].set_ylim(0.87, 0.93)
axes[2].set_ylabel("Test AUROC")
axes[2].set_title("Test Set AUROC", fontweight="bold")

for ax in axes:
    ax.tick_params(axis="x", labelsize=9)

legend_patches = [mpatches.Patch(color=c, label=m["modalities"])
                  for c, m in zip(colors, models_info)]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=9.5, frameon=True, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "nb_model_comparison.png"), dpi=130, bbox_inches="tight")
plt.show()

print()
print(f"{'Model':<25} {'Params':>8}  {'Epoch':>6}  {'Val F1':>7}  {'Test F1':>7}  {'AUROC':>7}")
print("-" * 65)
for m in models_info:
    print(f"{m['name'].replace(chr(10),' '):<25} {m['params']:>8,}  "
          f"{str(m['epoch']):>6}  {m['val_f1']:>7.4f}  "
          f"{m['test_f1']:>7.4f}  {m['test_auroc']:>7.4f}")
""")

# ------------------------------------------------------------------------------
md("## 12 - Early Fusion V2 -- Detailed Test Set Metrics")
code("""\
metrics_data = {
    "F1-score"  : 0.9003,
    "Precision" : 0.9368,
    "Recall"    : 0.8665,
    "AUROC"     : 0.8976,
}

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(list(metrics_data.keys()), list(metrics_data.values()),
               color=[C_BLUE, C_GREEN, C_ORANGE, C_ACCENT],
               edgecolor=C_MID, linewidth=0.7, height=0.5)

for bar, val in zip(bars, metrics_data.values()):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=12,
            fontweight="bold", color=C_MID)

ax.set_xlim(0.82, 0.97)
ax.set_xlabel("Score", fontsize=12)
ax.set_title("Early Fusion V2 -- Test Set Performance\\n"
             "threshold=0.55  |  epoch=2  |  val F1=0.9455",
             fontweight="bold", color=C_MID, fontsize=12)
ax.axvline(0.90, color=C_ACCENT, lw=1.3, linestyle=":", alpha=0.7)
ax.text(0.902, 0.05, "0.90", color=C_ACCENT, fontsize=9,
        transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "nb_ef2_metrics.png"), dpi=130, bbox_inches="tight")
plt.show()
""")

# ------------------------------------------------------------------------------
md("## 13 - Pre-saved Training Artefacts (Training Curves and ROC)")
code("""\
artefacts = [
    ("ef2_training_curves.png",          "EF-v2 Training Curves"),
    ("ef2_roc_curve.png",                "EF-v2 ROC Curve (test set)"),
    ("ef2_confusion_matrix.png",         "EF-v2 Confusion Matrix"),
    ("early_fusion_training_curves.png", "EF-v1 Training Curves"),
    ("training_curves.png",              "Kinematic LSTM Training Curves"),
]

for fname, title in artefacts:
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {fname}")
        continue
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontweight="bold", color=C_MID, fontsize=12)
    plt.tight_layout()
    plt.show()
    print(f"  Displayed: {fname}")
""")

# ------------------------------------------------------------------------------
md("""\
## 14 - Summary Table

| Item | Value |
|------|-------|
| Total frames | 17,559 |
| Event frames | 14,727 (83.9%) |
| Normal frames | 2,832 (16.1%) |
| Total episodes | 50 |
| Event clips | 68 |
| Test episodes | 9, 10, 47, 49 |
| Val episodes | 3, 6, 45, 46 |
| BEV grid | 64x64 px, 1 m/px, X in [-32, +32] m, Y in [-16, +48] m |
| Cache files | 17,559 .npz files, approx 527 MB |
| Best model | Early Fusion V2 -- epoch 2 |
| Best val F1 | 0.9455 |
| Test F1 | 0.9003 |
| Test Precision | 0.9368 |
| Test Recall | 0.8665 |
| Test AUROC | 0.8976 |
""")
code("""\
print("Sections completed:")
sections = [
    " 1  Annotation scan        -- 17,559 frames loaded",
    " 2  Dataset overview       -- 4-panel figure",
    " 3  Class balance          -- per-split bar chart",
    " 4  BEV LiDAR grid         -- channel views + event/normal comparison",
    " 5  RGB camera views       -- 6-view surround grid",
    " 6  Semantic views         -- 6-view surround grid",
    " 7  LiDAR point cloud      -- top-down, side-view, histogram",
    " 8  Kinematic distributions -- 6-panel feature histograms",
    " 9  Temporal patterns      -- aligned event timeline",
    "10  Cache statistics       -- feature norm and BEV occupancy",
    "11  Model comparison       -- 3 bar charts",
    "12  EF-v2 metrics          -- precision/recall/F1/AUROC",
    "13  Training artefacts     -- pre-saved curve and ROC plots",
    "14  Summary table",
    "15  Gallery (next cell)    -- all saved nb_*.png files",
]
for s in sections:
    print(" ", s)
""")

# ------------------------------------------------------------------------------
md("""\
## 15 - Gallery -- All Saved Plots

Every plot generated by this notebook is displayed below in one place.
Files are read from `models/nb_*.png`.
""")
code("""\
saved_plots = sorted(glob.glob(os.path.join(MODEL_DIR, "nb_*.png")))

if not saved_plots:
    print("No nb_*.png files found -- run the sections above first.")
else:
    print(f"Found {len(saved_plots)} saved plots:\\n")
    for img_path in saved_plots:
        fname = os.path.basename(img_path)
        size_kb = os.path.getsize(img_path) // 1024
        # Clean title: remove nb_ prefix and .png suffix, replace _ with spaces
        title = fname.replace("nb_", "").replace(".png", "").replace("_", " ").title()
        print(f"  {fname}  ({size_kb} KB)")
        display(IPImage(filename=img_path))
        print()
""")

# ==============================================================================
nb.cells = cells
nb.metadata = nbf.from_dict({
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.8.0",
    },
})

nbf.write(nb, OUT)
print(f"Written : {OUT}")
print(f"Cells   : {len(cells)}  ({sum(1 for c in cells if c.cell_type=='code')} code, "
      f"{sum(1 for c in cells if c.cell_type=='markdown')} markdown)")
