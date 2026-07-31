"""
Generates sensor_setup_diagram.png
A technical illustration of the ego vehicle sensor suite used in the project,
showing top-down coverage, side profile, and data flow.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Wedge, FancyBboxPatch, Circle, Arc
from matplotlib.gridspec import GridSpec
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUT     = os.path.join(OUT_DIR, "sensor_setup_diagram.png")

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BG     = "#0d1117"   # dark background
C_CAR    = "#21262d"   # car body fill
C_CAR_E  = "#58a6ff"   # car body edge
C_ROAD   = "#161b22"   # road surface
C_RGB    = "#58a6ff"   # RGB camera colour
C_SEM    = "#3fb950"   # Semantic camera colour
C_LIDAR  = "#e3b341"   # LiDAR colour
C_KIN    = "#f78166"   # Kinematics / GPS-IMU colour
C_BEV    = "#388bfd22" # BEV grid fill (transparent)
C_BEV_E  = "#388bfd"   # BEV grid edge
C_WHITE  = "#e6edf3"
C_GREY   = "#8b949e"
C_DARK   = "#0f3460"
C_ACCENT = "#e94560"

fig = plt.figure(figsize=(20, 14), facecolor=C_BG)
fig.patch.set_facecolor(C_BG)

gs = GridSpec(
    2, 3,
    figure=fig,
    left=0.04, right=0.97,
    top=0.92, bottom=0.04,
    hspace=0.38, wspace=0.32,
)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 (large, spans top row): Top-down sensor coverage
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(C_ROAD)
ax1.set_aspect("equal")
ax1.set_xlim(-55, 55)
ax1.set_ylim(-30, 62)
ax1.set_title("Top-Down Sensor Coverage Map  (ego vehicle at centre)",
              color=C_WHITE, fontsize=14, fontweight="bold", pad=10)
ax1.tick_params(colors=C_GREY)
for sp in ax1.spines.values():
    sp.set_edgecolor(C_GREY); sp.set_linewidth(0.5)
ax1.set_xlabel("X  [m]  lateral", color=C_GREY, fontsize=9)
ax1.set_ylabel("Y  [m]  longitudinal (forward = up)", color=C_GREY, fontsize=9)

# ── Grid lines ────────────────────────────────────────────────────────────────
for x in range(-50, 55, 10):
    ax1.axvline(x, color="#1f2937", linewidth=0.4, zorder=0)
for y in range(-30, 65, 10):
    ax1.axhline(y, color="#1f2937", linewidth=0.4, zorder=0)

# ── BEV grid extent ───────────────────────────────────────────────────────────
bev_rect = mpatches.FancyBboxPatch(
    (-32, -16), 64, 64,
    boxstyle="round,pad=0", linewidth=1.8,
    edgecolor=C_BEV_E, facecolor=C_BEV, zorder=1,
    linestyle="--",
)
ax1.add_patch(bev_rect)
ax1.text(-31, 47.5, "BEV Grid  64x64 px  |  1 m/px", color=C_BEV_E,
         fontsize=8.5, fontstyle="italic", zorder=10)
ax1.text(-31, 45.0, "X in [-32, +32] m   Y in [-16, +48] m", color=C_BEV_E,
         fontsize=8, zorder=10)

# ── Range rings ───────────────────────────────────────────────────────────────
for r, lbl in [(20, "20 m"), (40, "40 m"), (55, "55 m")]:
    ax1.add_patch(Circle((0, 0), r, fill=False,
                         edgecolor="#2d3748", linewidth=0.6,
                         linestyle=":", zorder=1))
    ax1.text(r * 0.707 + 0.5, r * 0.707 + 0.5, lbl,
             color=C_GREY, fontsize=7.5, zorder=10)

# ── LiDAR 360-degree coverage ─────────────────────────────────────────────────
lidar_range = 50
ax1.add_patch(Circle((0, 0), lidar_range, fill=True,
                     facecolor="#e3b34108", edgecolor=C_LIDAR,
                     linewidth=1.4, zorder=2))
# Radial lines at 60-degree intervals to show the 6 depth renders
for angle_deg in range(0, 360, 60):
    angle_rad = np.radians(angle_deg)
    ax1.plot([0, lidar_range * np.sin(angle_rad)],
             [0, lidar_range * np.cos(angle_rad)],
             color=C_LIDAR, linewidth=0.5, linestyle=":", alpha=0.5, zorder=3)

# ── Camera FOV wedges ─────────────────────────────────────────────────────────
# In matplotlib Wedge: angles are in degrees, 0=right, CCW positive
# Our convention: 0=up (forward), CW positive -> convert: matplotlib_angle = 90 - our_angle

CAM_FOV   = 65    # degrees full FOV per camera
CAM_RANGE = 45    # metres

# (name, position_xy, heading_deg[0=forward,CW], colour, real_world_name)
CAMERAS = [
    ("front",       ( 0.0,  2.1),    0,   C_RGB, "Front Camera"),
    ("front_right", ( 1.0,  1.5),   60,   C_RGB, "Front-Right Camera"),
    ("back_right",  ( 1.0, -1.5),  120,   C_RGB, "Back-Right Camera"),
    ("back",        ( 0.0, -2.1),  180,   C_RGB, "Back Camera"),
    ("back_left",   (-1.0, -1.5),  240,   C_RGB, "Back-Left Camera"),
    ("front_left",  (-1.0,  1.5),  300,   C_RGB, "Front-Left Camera"),
]

for name, pos, hdg, colour, rw_name in CAMERAS:
    # Convert our heading (0=forward/up, CW) to matplotlib (0=right, CCW)
    mpl_center = 90 - hdg
    w = Wedge(
        center=pos,
        r=CAM_RANGE,
        theta1=mpl_center - CAM_FOV / 2,
        theta2=mpl_center + CAM_FOV / 2,
        facecolor=colour + "18",   # very transparent fill
        edgecolor=colour,
        linewidth=1.2,
        zorder=4,
    )
    ax1.add_patch(w)

    # Camera dot
    ax1.plot(*pos, "o", color=colour, markersize=7, zorder=8,
             markeredgecolor="white", markeredgewidth=0.7)

    # Label offset along the camera direction
    off_r = CAM_RANGE * 0.55
    off_x = pos[0] + off_r * np.sin(np.radians(hdg))
    off_y = pos[1] + off_r * np.cos(np.radians(hdg))
    ax1.text(off_x, off_y, name.replace("_", "\n"),
             color=colour, fontsize=7.5, ha="center", va="center",
             fontweight="bold", zorder=10,
             bbox=dict(facecolor=C_BG, edgecolor="none", alpha=0.65, pad=1.5))

# ── Semantic camera note (same positions, different label) ────────────────────
ax1.text(0, -26,
         "Semantic segmentation views share the same 6 camera positions (computed in software)",
         color=C_SEM, fontsize=8.5, ha="center", zorder=10, fontstyle="italic")

# ── Car body ──────────────────────────────────────────────────────────────────
# Body
car_body = mpatches.FancyBboxPatch(
    (-1.0, -2.0), 2.0, 4.0,
    boxstyle="round,pad=0.15",
    facecolor=C_CAR, edgecolor=C_CAR_E, linewidth=2.0, zorder=7,
)
ax1.add_patch(car_body)

# Windscreen lines (front and rear)
for y_ws, col in [(1.3, "#90caf9"), (-1.3, "#546e7a")]:
    ax1.plot([-0.7, 0.7], [y_ws, y_ws], color=col, linewidth=1.5, zorder=9)

# Wheels
for wx, wy in [(-1.05, 1.2), (1.05, 1.2), (-1.05, -1.2), (1.05, -1.2)]:
    ax1.add_patch(mpatches.FancyBboxPatch(
        (wx - 0.15, wy - 0.35), 0.3, 0.7,
        boxstyle="round,pad=0.05",
        facecolor="#111", edgecolor=C_GREY, linewidth=1.0, zorder=9,
    ))

# LiDAR spinner on roof
ax1.add_patch(Circle((0, 0), 0.4, facecolor=C_LIDAR, edgecolor="white",
                     linewidth=1.0, zorder=10))
ax1.text(0, 0, "LiDAR", color="black", fontsize=6, ha="center", va="center",
         fontweight="bold", zorder=11)

# GPS/IMU dot (front-left of roof)
ax1.plot(-0.5, 0.8, "D", color=C_KIN, markersize=6, zorder=10,
         markeredgecolor="white", markeredgewidth=0.6)
ax1.text(-0.5, 0.8, "  GPS\n  IMU", color=C_KIN, fontsize=6.5,
         va="center", zorder=11)

# Forward arrow
ax1.annotate("", xy=(0, 5), xytext=(0, 2.5),
             arrowprops=dict(arrowstyle="-|>", color=C_WHITE,
                             lw=1.5, mutation_scale=12), zorder=10)
ax1.text(0.3, 4.0, "Forward", color=C_WHITE, fontsize=8, va="center", zorder=10)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_RGB   + "30", edgecolor=C_RGB,   label="RGB Camera FOV  (x6, 65 deg each)"),
    mpatches.Patch(facecolor=C_LIDAR + "15", edgecolor=C_LIDAR, label="LiDAR 360 deg coverage  (50 m range)"),
    mpatches.Patch(facecolor=C_BEV,          edgecolor=C_BEV_E, label="BEV Grid extent  64x64 m",
                   linestyle="--"),
    mpatches.Patch(facecolor="none",         edgecolor=C_KIN,
                   label="GPS + IMU  (ego kinematics)"),
]
ax1.legend(handles=legend_items, loc="lower right",
           facecolor=C_CAR, edgecolor=C_GREY,
           labelcolor=C_WHITE, fontsize=9, framealpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 (bottom-left): Side profile view
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(C_BG)
ax2.set_aspect("equal")
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-0.6, 3.2)
ax2.set_title("Side Profile  (right-hand view)", color=C_WHITE,
              fontsize=11, fontweight="bold", pad=8)
ax2.axis("off")

# Road surface
ax2.axhline(0, color=C_GREY, linewidth=1.0)
ax2.fill_between([-3.5, 3.5], [-0.6, -0.6], [0, 0], color="#161b22")

# Wheels
for wx in [-1.8, 1.8]:
    ax2.add_patch(Circle((wx, 0.35), 0.35, facecolor="#222",
                         edgecolor=C_GREY, linewidth=1.2))
    ax2.add_patch(Circle((wx, 0.35), 0.15, facecolor=C_GREY,
                         edgecolor=C_BG, linewidth=0.8))

# Car body silhouette (simplified sedan shape)
body_x = [-2.2, -2.2, -2.0, -1.2, -0.6,  0.6,  1.5,  2.2,  2.2]
body_y = [ 0.7,  1.2,  1.35, 1.7,  1.85, 1.85,  1.55, 1.2,  0.7]
ax2.fill(body_x + [body_x[-1], body_x[0]],
         body_y + [0.7, 0.7],
         facecolor=C_CAR, edgecolor=C_CAR_E, linewidth=1.8, zorder=3)

# Windscreen
ax2.plot([-1.2, -0.6], [1.7, 1.85], color="#90caf9", linewidth=2.0, zorder=4)
ax2.plot([ 1.5,  0.6], [1.55, 1.85], color="#546e7a", linewidth=2.0, zorder=4)

# LiDAR on roof
lidar_y = 1.95
ax2.add_patch(Circle((0, lidar_y), 0.18, facecolor=C_LIDAR,
                     edgecolor="white", linewidth=1.0, zorder=5))
# Rotation arcs
for a in [30, 90, 150, 210, 270, 330]:
    r = 0.14
    ax2.plot(r * np.cos(np.radians(a)) + 0,
             r * np.sin(np.radians(a)) + lidar_y,
             ".", color="black", markersize=2, zorder=6)
ax2.annotate("LiDAR\n(spinning, 360 deg)", xy=(0, lidar_y + 0.2),
             xytext=(1.6, 2.7),
             arrowprops=dict(arrowstyle="-|>", color=C_LIDAR, lw=1.2),
             color=C_LIDAR, fontsize=8, fontweight="bold", zorder=7)

# GPS/IMU
gps_pos = (-0.5, lidar_y)
ax2.add_patch(mpatches.FancyBboxPatch(
    (gps_pos[0] - 0.08, gps_pos[1] - 0.05), 0.16, 0.10,
    boxstyle="round,pad=0.02", facecolor=C_KIN, edgecolor="white",
    linewidth=0.8, zorder=5))
ax2.annotate("GPS + IMU\n(ego state)", xy=gps_pos,
             xytext=(-2.5, 2.7),
             arrowprops=dict(arrowstyle="-|>", color=C_KIN, lw=1.2),
             color=C_KIN, fontsize=8, fontweight="bold", zorder=7)

# Front camera
fc_pos = (2.22, 1.1)
ax2.plot(*fc_pos, "o", color=C_RGB, markersize=8, zorder=5,
         markeredgecolor="white", markeredgewidth=0.7)
ax2.annotate("Front Camera\n(+5 others around car)", xy=fc_pos,
             xytext=(2.0, 2.5),
             arrowprops=dict(arrowstyle="-|>", color=C_RGB, lw=1.2),
             color=C_RGB, fontsize=8, fontweight="bold", zorder=7)

# Camera FOV cone (front)
fov_half = np.radians(32)
for sign in [1, -1]:
    ax2.plot([fc_pos[0], fc_pos[0] + 1.8 * np.cos(sign * fov_half)],
             [fc_pos[1], fc_pos[1] + 1.8 * np.sin(sign * fov_half)],
             color=C_RGB, linewidth=0.8, linestyle="--", alpha=0.7, zorder=4)

# Ground label
ax2.text(0, -0.35, "Road surface", color=C_GREY, fontsize=8,
         ha="center", fontstyle="italic")

# Dimensions
ax2.annotate("", xy=(-2.2, 0.0), xytext=(2.2, 0.0),
             arrowprops=dict(arrowstyle="<->", color=C_GREY, lw=0.8))
ax2.text(0, -0.15, "~4.5 m", color=C_GREY, fontsize=7.5, ha="center")

ax2.annotate("", xy=(3.1, 0), xytext=(3.1, 1.9),
             arrowprops=dict(arrowstyle="<->", color=C_GREY, lw=0.8))
ax2.text(3.2, 0.95, "~1.9 m", color=C_GREY, fontsize=7.5, va="center")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 (bottom-middle): Sensor to data pipeline
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(C_BG)
ax3.axis("off")
ax3.set_title("Sensor -> Data -> Model Input Pipeline",
              color=C_WHITE, fontsize=11, fontweight="bold", pad=8)

def box(ax, xy, w, h, label, sublabel, colour, fontsize=9):
    rect = FancyBboxPatch(
        (xy[0] - w/2, xy[1] - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=colour + "22", edgecolor=colour, linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(xy[0], xy[1] + 0.03, label, color=colour,
            ha="center", va="center", fontsize=fontsize, fontweight="bold")
    if sublabel:
        ax.text(xy[0], xy[1] - 0.10, sublabel, color=C_GREY,
                ha="center", va="center", fontsize=fontsize - 1.5)

def arrow(ax, x1, y1, x2, y2, colour=C_GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=colour,
                                lw=1.2, mutation_scale=10))

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# Row 1: Real-world sensors
sensor_y = 0.88
sensors = [
    (0.18, C_RGB,   "6x Cameras",       "RGB   640x360"),
    (0.50, C_SEM,   "Segmentation",      "software  on RGB"),
    (0.82, C_LIDAR, "LiDAR",             "spinning  laser"),
]
for sx, col, lbl, sub in sensors:
    box(ax3, (sx, sensor_y), 0.28, 0.13, lbl, sub, col)

# GPS/IMU row
box(ax3, (0.50, 0.70), 0.28, 0.10, "GPS + IMU", "ego state", C_KIN, fontsize=8)

# Arrows down
for sx, col, *_ in sensors:
    arrow(ax3, sx, sensor_y - 0.07, sx, 0.57, col)
arrow(ax3, 0.50, 0.65, 0.50, 0.57, C_KIN)

# Row 2: Processing step
proc_y = 0.50
procs = [
    (0.18, C_RGB,   "MobileNetV3\n(frozen)",  "(576-dim vector)"),
    (0.50, C_SEM,   "MobileNetV3\n(frozen)",  "(576-dim vector)"),
    (0.82, C_LIDAR, "BEV Grid\nprojection",   "(3 x 64 x 64)"),
]
for px, col, lbl, sub in procs:
    box(ax3, (px, proc_y), 0.28, 0.13, lbl, sub, col)
box(ax3, (0.50, 0.36), 0.28, 0.10, "Kinematic\nextraction", "(19-dim vector)", C_KIN, fontsize=8)

# Arrows down
for px, col, *_ in procs:
    arrow(ax3, px, proc_y - 0.07, px, 0.24, col)
arrow(ax3, 0.50, 0.31, 0.50, 0.24, C_KIN)

# Row 3: Projection heads
proj_y = 0.18
projs = [
    (0.12, C_RGB,   "RGB proj", "-> 64-dim"),
    (0.38, C_SEM,   "Sem proj", "-> 32-dim"),
    (0.62, C_LIDAR, "BEV CNN",  "-> 64-dim"),
    (0.88, C_KIN,   "JSON proj","-> 32-dim"),
]
for px, col, lbl, sub in projs:
    box(ax3, (px, proj_y), 0.22, 0.10, lbl, sub, col, fontsize=8)

# Arrows down
for px, col, *_ in projs:
    arrow(ax3, px, proj_y - 0.06, 0.50, 0.06, col)

# Final fusion box
box(ax3, (0.50, 0.04), 0.50, 0.07,
    "Concatenate  ->  192-dim fused vector",
    None, C_WHITE, fontsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 (bottom-right): Sensor specs table
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor(C_BG)
ax4.axis("off")
ax4.set_title("Sensor Specifications",
              color=C_WHITE, fontsize=11, fontweight="bold", pad=8)

rows = [
    # (Sensor, Simulation, Real-world equivalent, Dim)
    ("Sensor",       "Simulation",          "Real-World Equivalent",   "Model Input"),
    ("RGB Camera",   "6 rendered views\n640x360 px, 10 FPS",
                     "Physical CMOS cameras\n(20-30 FPS, wide-angle)",
                     "576-dim vector\n(MobileNetV3)"),
    ("Semantic\nCamera",  "MetaDrive pixel\nlabels (free)",
                     "DeepLabV3 / SegFormer\nrun on RGB frame",
                     "576-dim vector\n(MobileNetV3)"),
    ("LiDAR",        "6 depth renders\n@ 60 deg intervals",
                     "Spinning laser scanner\n10-20 rev/sec, 100K pts/sec",
                     "3x64x64 BEV grid\n(float16 in RAM)"),
    ("Kinematics /\nGPS + IMU",  "JSON annotation\nfrom simulator",
                     "GPS + IMU (ego)\nRADAR + Kalman (NPCs)",
                     "19-dim vector\n(raw numbers)"),
    ("Event Label",  "Exact trigger frame\n(from simulator code)",
                     "Human annotator\n(+-1-2 sec precision)",
                     "Binary 0/1\nper frame"),
]

col_w = [0.20, 0.25, 0.30, 0.25]
col_x = [0.00, 0.20, 0.45, 0.75]
row_h = 0.145
colours_row = [C_WHITE, C_RGB, C_SEM, C_LIDAR, C_KIN, C_ACCENT]

for ri, (row, row_colour) in enumerate(zip(rows, colours_row)):
    y_pos = 0.95 - ri * row_h
    for ci, (cell, cx) in enumerate(zip(row, col_x)):
        is_header = (ri == 0)
        bg = "#1a2233" if (ri % 2 == 0 and ri > 0) else C_BG
        if is_header:
            bg = "#1e3a5f"
        ax4.text(
            cx + col_w[ci] / 2,
            y_pos,
            cell,
            color=row_colour if not is_header else C_WHITE,
            fontsize=7.5,
            ha="center",
            va="top",
            fontweight="bold" if is_header else "normal",
            transform=ax4.transAxes,
            wrap=True,
        )
        # cell border
        rect = FancyBboxPatch(
            (cx, y_pos - row_h + 0.005), col_w[ci] - 0.005, row_h - 0.005,
            boxstyle="round,pad=0.005",
            facecolor=bg, edgecolor="#2d3748",
            linewidth=0.5, transform=ax4.transAxes, zorder=0,
        )
        ax4.add_patch(rect)

# Key note at bottom
ax4.text(0.5, 0.02,
         "* In simulation the event label is frame-perfect.\n"
         "  In reality, human annotators introduce +/-10-20 frame uncertainty.",
         color=C_GREY, fontsize=7.5, ha="center", va="bottom",
         transform=ax4.transAxes, fontstyle="italic")

# ══════════════════════════════════════════════════════════════════════════════
# Master title
# ══════════════════════════════════════════════════════════════════════════════
fig.text(
    0.5, 0.965,
    "Ego Vehicle Sensor Suite -- MetaDrive Simulation vs Real-World AV",
    color=C_WHITE, fontsize=16, fontweight="bold", ha="center", va="top",
    fontfamily="monospace",
)
fig.text(
    0.5, 0.945,
    "Project: Simulation-Based Rash Driving Behaviour Prediction  |  "
    "Kyutech 2026",
    color=C_GREY, fontsize=10, ha="center", va="top",
)

plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {OUT}")
print(f"Size : {os.path.getsize(OUT) // 1024} KB")
