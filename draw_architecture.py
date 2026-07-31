"""
draw_architecture.py
Generates models/architecture_ef2.png — Early Fusion V2 architecture diagram.
Run: python draw_architecture.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 22, 16
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
BG = '#F0F4F8'
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BEV    = '#1D4ED8'   # blue-700
C_RGB    = '#15803D'   # green-700
C_SEM    = '#B45309'   # amber-700
C_KIN    = '#6D28D9'   # violet-700
C_FROZEN = '#6B7280'   # gray-500   (frozen modules)
C_FUSE   = '#1E3A5F'   # navy
C_GRU    = '#78350F'   # amber-900  (brown)
C_HEAD   = '#1F2937'   # gray-800
C_OUT    = '#991B1B'   # red-800
C_INBG   = '#DBEAFE'   # input box fill
C_INTXT  = '#1E3A5F'   # input box text

STREAM_C = [C_BEV, C_RGB, C_SEM, C_KIN]
FROZEN_F = [False,  True,  True, False]

# ── Helper: rounded box ───────────────────────────────────────────────────────
def rbox(cx, cy, w, h, text, fc, tc='white', fs=8.5, frozen=False, z=4, bold=True):
    pat = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.12',
        facecolor=fc, edgecolor='white',
        linewidth=1.6 if not frozen else 1.0,
        linestyle='--' if frozen else '-',
        alpha=0.93, zorder=z
    )
    ax.add_patch(pat)
    ax.text(cx, cy, text,
            ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal',
            color=tc, zorder=z + 1,
            multialignment='center', linespacing=1.4)

# ── Helper: arrow ─────────────────────────────────────────────────────────────
def arr(x1, y1, x2, y2, c='#555555', lw=1.7):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=c, lw=lw, mutation_scale=14
                ), zorder=9)

# ── Helper: plain line ────────────────────────────────────────────────────────
def line(xs, ys, c='#555555', lw=1.7, z=8):
    ax.plot(xs, ys, color=c, lw=lw, zorder=z)

# ── Layout constants ──────────────────────────────────────────────────────────
XS   = [2.75, 7.25, 13.25, 18.75]     # 4 modality column x-centres
XCTR = (XS[0] + XS[-1]) / 2           # = 10.75

# y-levels (top → bottom)
Y_TITLE    = 15.60
Y_SUB      = 15.12
Y_SECLBL   = 14.62
Y_COLNAME  = 14.15
Y_IN       = 13.48   # input data boxes
Y_BB       = 12.05   # backbone / BEV CNN
Y_PROJ     = 10.65   # projection layers
Y_DIM      =  9.38   # output-dim tag boxes
Y_BAR      =  8.65   # horizontal convergence bar
Y_CAT      =  7.90   # concat box centre
Y_TEMP     =  6.92   # temporal stacking note
Y_GRU      =  5.82   # BiGRU
Y_H1       =  4.72   # head block 1
Y_H2       =  3.80   # head block 2
Y_OUT      =  2.85   # output box

# ── Per-frame background region ───────────────────────────────────────────────
bg_bot = Y_DIM - 0.55
bg_top = Y_SECLBL + 0.40
bg = FancyBboxPatch(
    (0.3, bg_bot), 21.4, bg_top - bg_bot,
    boxstyle='round,pad=0.2',
    facecolor='#E0EDFF', edgecolor='#1D4ED8',
    linewidth=1.5, alpha=0.40, zorder=1
)
ax.add_patch(bg)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(XCTR, Y_TITLE,
        'Early Fusion V2  —  Architecture',
        ha='center', va='center',
        fontsize=18, fontweight='bold', color='#1E3A5F')
ax.text(XCTR, Y_SUB,
        'Simulation-Based Rash Driving Behaviour Prediction  ·  Kyushu Institute of Technology',
        ha='center', va='center',
        fontsize=10, color='#4B5563', style='italic')

# Section label
ax.text(XCTR, Y_SECLBL,
        '─────   Per-Frame Feature Extraction   '
        '(applied at each of  T = 10  frames)   ─────',
        ha='center', va='center',
        fontsize=9, color='#1D4ED8', fontweight='bold')

# ── Column labels ─────────────────────────────────────────────────────────────
COL_NAMES  = ['BEV  LiDAR',
              'RGB  Cameras',
              'Semantic  Cameras',
              'Kinematics  (JSON)']
for x, name, c in zip(XS, COL_NAMES, STREAM_C):
    ax.text(x, Y_COLNAME, name,
            ha='center', va='center',
            fontsize=10, fontweight='bold', color=c)

# ── ROW 1 — Input data boxes ──────────────────────────────────────────────────
INPUT_TEXT = [
    '3 × 64 × 64\nBEV grid',
    '6 × RGB\n640 × 360 px',
    '6 × Semantic\n640 × 360 px',
    '19-dim\nkinematic vector',
]
for x, txt in zip(XS, INPUT_TEXT):
    rbox(x, Y_IN, 3.7, 0.80, txt,
         fc=C_INBG, tc=C_INTXT, fs=9, z=3, bold=False)

# ── ROW 2 — Backbone / BEV CNN ───────────────────────────────────────────────
BB_TEXT = [
    'BEV CNN  (trainable)\nConv×3 + BN + ReLU\nAdaptiveAvgPool(4×4)',
    'MobileNetV3-Small\n(frozen · ImageNet)\n576-dim output / view',
    'MobileNetV3-Small\n(frozen · ImageNet)\n576-dim output / view',
    'Linear(19 → 32)\n+ LayerNorm + ReLU',
]
for x, txt, c, frz in zip(XS, BB_TEXT, STREAM_C, FROZEN_F):
    arr(x, Y_IN  - 0.40, x, Y_BB + 0.56)
    fc = C_FROZEN if frz else c
    tc = '#E5E7EB' if frz else 'white'
    rbox(x, Y_BB, 3.7, 1.08, txt, fc=fc, tc=tc, fs=8, frozen=frz)
    if frz:
        ax.text(x, Y_BB + 0.66, '❄  frozen — no gradient',
                ha='center', va='bottom',
                fontsize=7.5, color='#6B7280', style='italic')

# ── ROW 3 — Projection layers ─────────────────────────────────────────────────
PROJ_TEXT = [
    'Flatten\nLinear(1024 → 64) + ReLU\nDropout(0.50)',
    'Mean-pool 6 views\nLinear(576 → 64)\n+ LayerNorm + ReLU',
    'Mean-pool 6 views\nLinear(576 → 32)\n+ LayerNorm + ReLU',
    None,   # kinematics — no second block
]
for i, (x, c) in enumerate(zip(XS, STREAM_C)):
    if PROJ_TEXT[i]:
        arr(x, Y_BB - 0.54, x, Y_PROJ + 0.54)
        rbox(x, Y_PROJ, 3.7, 1.04, PROJ_TEXT[i], fc=c, fs=8)

# ── ROW 4 — Output-dimension tags ────────────────────────────────────────────
DIM_TEXT = [
    '64-dim  spatial feature',
    '64-dim  visual feature',
    '32-dim  scene feature',
    '32-dim  motion feature',
]
for i, (x, c, dt) in enumerate(zip(XS, STREAM_C, DIM_TEXT)):
    src_y = (Y_PROJ - 0.52) if PROJ_TEXT[i] else (Y_BB - 0.54)
    arr(x, src_y, x, Y_DIM + 0.31)
    rbox(x, Y_DIM, 3.5, 0.58, dt, fc=c, fs=8.5)

# ── Convergence: dim tags → concat ────────────────────────────────────────────
# Vertical lines down from each tag to horizontal bar
for x in XS:
    line([x, x], [Y_DIM - 0.29, Y_BAR])
# Horizontal bar connecting the four verticals
line([XS[0], XS[-1]], [Y_BAR, Y_BAR], lw=1.9)
# Single downward arrow from bar centre to concat
arr(XCTR, Y_BAR, XCTR, Y_CAT + 0.38)

# ── Concat box ────────────────────────────────────────────────────────────────
rbox(XCTR, Y_CAT, 14.5, 0.72,
     'Concatenate  →  192-dim  fused feature vector  (per frame)',
     fc=C_FUSE, fs=11)

# ── Temporal stacking note ────────────────────────────────────────────────────
arr(XCTR, Y_CAT - 0.36, XCTR, Y_TEMP + 0.30)
ax.text(XCTR, Y_TEMP,
        'Stack  T = 10  frames    →    Sequence  (batch × 10 × 192)',
        ha='center', va='center',
        fontsize=10, fontweight='bold', color='#1E3A5F',
        bbox=dict(boxstyle='round,pad=0.38',
                  facecolor='#DBEAFE', edgecolor='#1D4ED8', lw=1.6),
        zorder=6)

# ── BiGRU ─────────────────────────────────────────────────────────────────────
arr(XCTR, Y_TEMP - 0.28, XCTR, Y_GRU + 0.46)
rbox(XCTR, Y_GRU, 15.0, 0.88,
     'BiGRU   |   hidden = 96,   2 layers,   bidirectional,   dropout = 0.50\n'
     'Output: 192-dim  (2 × 96)  hidden state at final timestep',
     fc=C_GRU, fs=9.5)

# ── Classification head — block 1 ─────────────────────────────────────────────
arr(XCTR, Y_GRU - 0.44, XCTR, Y_H1 + 0.34)
rbox(XCTR, Y_H1, 13.5, 0.64,
     'Dropout(0.50)  →  Linear(192 → 64)  →  ReLU  →  Dropout(0.25)',
     fc=C_HEAD, fs=9.5)

# ── Classification head — block 2 ─────────────────────────────────────────────
arr(XCTR, Y_H1 - 0.32, XCTR, Y_H2 + 0.30)
rbox(XCTR, Y_H2, 7.0, 0.56,
     'Linear(64 → 1)  →  Sigmoid',
     fc=C_HEAD, fs=10.5)

# ── Output ────────────────────────────────────────────────────────────────────
arr(XCTR, Y_H2 - 0.28, XCTR, Y_OUT + 0.35, c=C_OUT, lw=2.8)
rbox(XCTR, Y_OUT, 10.5, 0.66,
     'P(rash event) ∈ [0, 1]    →    Alarm  if  P > 0.55',
     fc=C_OUT, fs=11.5)

# ── Legend ────────────────────────────────────────────────────────────────────
Lx, Ly = 0.45, 3.55
ax.text(Lx + 0.05, Ly + 0.08, 'Legend',
        fontsize=9, fontweight='bold', color='#1F2937')
LEGEND = [
    (C_BEV,    'BEV LiDAR  (trainable CNN)'),
    (C_RGB,    'RGB Cameras  (frozen backbone)'),
    (C_SEM,    'Semantic Cameras  (frozen backbone)'),
    (C_KIN,    'Kinematics / JSON'),
    (C_FROZEN, '– – –  Frozen  (no gradient flow)'),
]
for j, (c, lbl) in enumerate(LEGEND):
    yy = Ly - 0.40 - j * 0.50
    p = FancyBboxPatch((Lx, yy - 0.14), 0.40, 0.32,
                        boxstyle='round,pad=0.04',
                        facecolor=c, edgecolor='white',
                        linewidth=0.8, zorder=5)
    ax.add_patch(p)
    ax.text(Lx + 0.56, yy + 0.02, lbl,
            fontsize=8.5, va='center', color='#1F2937')

# ── Parameter count badge ─────────────────────────────────────────────────────
ax.text(20.4, 2.62,
        '492,193\ntotal params',
        ha='center', va='center',
        fontsize=10, fontweight='bold', color='#1E3A5F',
        bbox=dict(boxstyle='round,pad=0.55',
                  facecolor='#DBEAFE', edgecolor='#1D4ED8', lw=2.0),
        zorder=6)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'architecture_ef2.png')

plt.savefig(out_path, dpi=180, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print(f'Saved → {out_path}')
