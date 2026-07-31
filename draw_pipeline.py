"""
draw_pipeline.py
Generates models/pipeline_ef2.png  —  Slide 11: end-to-end system pipeline.
Run: python draw_pipeline.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 14, 20
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
BG = '#F0F4F8'
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

XCTR = FIG_W / 2   # 7.0

# ── Colours ───────────────────────────────────────────────────────────────────
C_SIM   = '#065F46'   # emerald dark   — simulator
C_DATA  = '#1E40AF'   # blue dark      — dataset
C_CACHE = '#5B21B6'   # violet dark    — feature cache
C_MODEL = '#1E3A5F'   # navy           — model outer
C_BEV   = '#1D4ED8'   # blue
C_RGB   = '#15803D'   # green
C_SEM   = '#B45309'   # amber
C_KIN   = '#6D28D9'   # violet
C_GRU   = '#78350F'   # brown
C_HEAD  = '#1F2937'   # dark gray
C_POUT  = '#7F1D1D'   # dark red       — P(rash) inside model
C_ALARM = '#991B1B'   # red            — alarm box

# ── Primitive helpers ─────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, text, fc, tc='white', fs=9.5, z=4, bold=True, alpha=0.93):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.15',
        facecolor=fc, edgecolor='white',
        linewidth=2.0, alpha=alpha, zorder=z))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal',
            color=tc, zorder=z + 1,
            multialignment='center', linespacing=1.4)

def outline_box(cx, cy, w, h, ec, fc='none', lw=2.2, alpha=0.75, z=2):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.15',
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=z))

def big_arrow(y1, y2, c='#374151', lw=2.4):
    ax.annotate('', xy=(XCTR, y2), xytext=(XCTR, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                mutation_scale=20), zorder=9)

def arrow_label(y, text, fs=8.5):
    ax.text(XCTR, y, text, ha='center', va='center',
            fontsize=fs, color='#1F2937', style='italic',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                      edgecolor='#94A3B8', lw=1.0, alpha=0.92),
            zorder=10)

def mini_arr(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#9CA3AF',
                                lw=1.3, mutation_scale=10), zorder=6)

def step_badge(y, number, color):
    """Numbered circle on the left margin."""
    circle = plt.Circle((0.62, y), 0.36, color=color, zorder=8, clip_on=False)
    ax.add_patch(circle)
    ax.text(0.62, y, str(number), ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=9)

# ── Inner-box x-centres (shared by dataset modal row & model proj row) ────────
BOX_W = 2.60
GAP   = 0.42
SPAN  = 4 * BOX_W + 3 * GAP
IXS   = (XCTR - SPAN / 2) + np.arange(4) * (BOX_W + GAP) + BOX_W / 2
# IXS ≈ [2.39, 5.41, 8.43, 11.45]

# ═════════════════════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════════════════════
ax.text(XCTR, 19.50, 'Proposed System  —  End-to-End Pipeline',
        ha='center', va='center',
        fontsize=16, fontweight='bold', color='#1E3A5F')
ax.text(XCTR, 19.05, 'Simulation  →  Dataset  →  Feature cache  →  Model  →  Alarm',
        ha='center', va='center',
        fontsize=10, color='#4B5563', style='italic')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — MetaDrive Simulator
# ═════════════════════════════════════════════════════════════════════════════
Y1 = 18.1
rbox(XCTR, Y1, 12.2, 1.0,
     'MetaDrive 0.4.3  Simulator\n'
     '50 episodes  ·  Procedural road generation  ·  Scripted rash-driving NPCs',
     fc=C_SIM, fs=10.5)
step_badge(Y1, 1, C_SIM)

# Arrow
big_arrow(Y1 - 0.50, 16.88, c='#374151')
arrow_label(17.22,
            'Scripted events: cut-in & emergency brake  ·  10 FPS  ·  Rolling 50-frame buffer capture')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Dataset Collection
# ═════════════════════════════════════════════════════════════════════════════
Y2_CTR = 16.08
Y2_H   = 1.74
# Light-filled outline box
ax.add_patch(FancyBboxPatch(
    (XCTR - 6.55, Y2_CTR - Y2_H / 2), 13.1, Y2_H,
    boxstyle='round,pad=0.15',
    facecolor='#EFF6FF', edgecolor=C_DATA,
    linewidth=2.2, alpha=0.80, zorder=2))

ax.text(XCTR, Y2_CTR + 0.67,
        'Dataset Collection   (17,559 frames  ·  68 event clips)',
        ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_DATA, zorder=4)
step_badge(Y2_CTR, 2, C_DATA)

# 4 modality sub-boxes
MODAL_C = [C_BEV, C_RGB, C_SEM, C_KIN]
MODAL_T = [
    'LiDAR\nPoint Cloud\n(6 depth views)',
    '6 × RGB\n640×360 px\nPNG',
    '6 × Semantic\n640×360 px\nPNG',
    'JSON\nAnnotations\n(19-dim)',
]
for ix, mc, mt in zip(IXS, MODAL_C, MODAL_T):
    rbox(ix, Y2_CTR - 0.12, BOX_W - 0.12, 1.02, mt, fc=mc, fs=8.2, z=5)

# Arrow
big_arrow(Y2_CTR - Y2_H / 2, 14.62, c='#374151')
arrow_label(14.97,
            'Frozen MobileNetV3-Small  ·  BEV grid generation  ·  Batched forward pass (5× speedup)')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Feature Cache
# ═════════════════════════════════════════════════════════════════════════════
Y3 = 14.15
rbox(XCTR, Y3, 11.8, 0.95,
     'Feature Cache   (527 MB  ·  17,559 .npz files)\n'
     'rgb_feat (576)  +  sem_feat (576)  +  bev (3×64×64, float16)',
     fc=C_CACHE, fs=10)
step_badge(Y3, 3, C_CACHE)

# Arrow
big_arrow(Y3 - 0.48, 12.88, c='#374151')
arrow_label(13.22,
            'Sliding 10-frame windows (1 second)  ·  Episode-level split  ·  Batch = 32')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Early Fusion V2 Model
# ═════════════════════════════════════════════════════════════════════════════
MODEL_BOT = 6.90
MODEL_TOP = 12.58
MODEL_CTR = (MODEL_BOT + MODEL_TOP) / 2   # 9.74

# Outer model box
ax.add_patch(FancyBboxPatch(
    (XCTR - 6.72, MODEL_BOT), 13.44, MODEL_TOP - MODEL_BOT,
    boxstyle='round,pad=0.18',
    facecolor='#E8EDF5', edgecolor=C_MODEL,
    linewidth=2.5, alpha=0.90, zorder=2))
ax.text(XCTR, MODEL_TOP - 0.38,
        'Early Fusion V2  —  Model   (492,193 params)',
        ha='center', va='center',
        fontsize=12, fontweight='bold', color=C_MODEL, zorder=4)
step_badge(MODEL_CTR, 4, C_MODEL)

# Internal y-levels (from top of model box down)
Y_PROJ  = MODEL_TOP - 1.28
Y_CONCAT = MODEL_TOP - 2.42
Y_GRU   = MODEL_TOP - 3.48
Y_HEAD  = MODEL_TOP - 4.38
Y_POUT  = MODEL_BOT + 0.42

# — 4 modality projection sub-boxes
PROJ_C = [C_BEV, C_RGB, C_SEM, C_KIN]
PROJ_T = [
    'BEV CNN\n→ 64-dim\nspatial feature',
    'RGB proj\nLinear(576→64)\n→ 64-dim',
    'Sem proj\nLinear(576→32)\n→ 32-dim',
    'Kin proj\nLinear(19→32)\n→ 32-dim',
]
for ix, pc, pt in zip(IXS, PROJ_C, PROJ_T):
    rbox(ix, Y_PROJ, BOX_W - 0.10, 1.02, pt, fc=pc, fs=7.8, z=5)
    mini_arr(ix, Y_PROJ - 0.51, ix, Y_CONCAT + 0.27)

# — Concat
rbox(XCTR, Y_CONCAT, 11.8, 0.50,
     'Concatenate   →   192-dim  fused feature vector  (per frame)',
     fc=C_MODEL, fs=9.5, z=5)
mini_arr(XCTR, Y_CONCAT - 0.25, XCTR, Y_GRU + 0.30)

# — BiGRU
rbox(XCTR, Y_GRU, 11.8, 0.56,
     'BiGRU   |   hidden=96,   2 layers,   bidirectional,   dropout=0.50',
     fc=C_GRU, fs=9.5, z=5)
mini_arr(XCTR, Y_GRU - 0.28, XCTR, Y_HEAD + 0.28)

# — Classification head
rbox(XCTR, Y_HEAD, 10.5, 0.52,
     'Dropout(0.5) → Linear(192→64) → ReLU → Linear(64→1) → Sigmoid',
     fc=C_HEAD, fs=9.0, z=5)
mini_arr(XCTR, Y_HEAD - 0.26, XCTR, Y_POUT + 0.26)

# — P(rash event) output (inside model box)
rbox(XCTR, Y_POUT, 8.0, 0.50,
     'P(rash event) ∈ [0, 1]     threshold = 0.55',
     fc=C_POUT, tc='#FEE2E2', fs=10, z=5)

# Arrow out of model
big_arrow(MODEL_BOT, 6.02, c=C_ALARM, lw=2.6)
arrow_label(6.42, 'Alarm raised if  P > 0.55')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Alarm
# ═════════════════════════════════════════════════════════════════════════════
Y5 = 5.52
rbox(XCTR, Y5, 12.2, 0.95,
     'ALARM  —  Rash behaviour predicted in current 1-second window\n'
     'Ego AV reacts:  adjust speed  ·  alert system  ·  handover signal',
     fc=C_ALARM, fs=10.5)
step_badge(Y5, 5, C_ALARM)

# ═════════════════════════════════════════════════════════════════════════════
# Results badge (bottom-right)
ax.text(12.6, 4.55,
        'Test results\nF1 = 0.90\nPrec = 0.94\nAUROC = 0.90',
        ha='center', va='center', fontsize=9, fontweight='bold', color='#1E3A5F',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#DBEAFE',
                  edgecolor='#1D4ED8', lw=1.8), zorder=6)

# ═════════════════════════════════════════════════════════════════════════════
# Save
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'pipeline_ef2.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print('Saved to: ' + out_path)
