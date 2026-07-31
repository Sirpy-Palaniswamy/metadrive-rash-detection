"""
MetaDrive Multi-Modal Fusion Dataset Collector  (v3 — corrected)
================================================================
Collects event-triggered sensor data suitable for early-fusion AV models.

Sensors
-------
- 6-camera surround RGB  (640×360, sequential render from 2 framebuffers)
- 6-camera surround Semantic  (same framebuffer trick)
- LiDAR point cloud  (N×4 float32 : X,Y,Z,intensity — ego-centric metres)

LiDAR fix (v3)
--------------
MetaDrive's PointCloudLidar.get_rgb_array_cpu() contains two bugs:
  1. cx/cy are swapped (uses BUFFER_H for cx and BUFFER_W for cy).
  2. Depth image is transposed before meshing, misaligning pixel→ray.
The result is point clouds in world-scale garbage coordinates (−13 000 m …
+18 000 m) instead of ego-centric ±100 m.
Fix: bypass get_rgb_array_cpu() entirely.  Access depth_tex (float32 [0,1])
directly, apply correct intrinsics, back-project in Panda3D camera coords
(Y-forward, X-right, Z-up), then add the mount offset.

Bounding-box fix (v3)
---------------------
All boxes are now stored in BOTH world frame and ego-centric frame.
Ego-centric = translate by (−ego_pos) then rotate by (−ego_heading) around Z.
LWH centre-Z is set to obj.HEIGHT/2 above ground (not hardcoded 0).

Calibration fix (v3)
--------------------
calibration.json now contains per-camera 4×4 T_cam_to_ego and
T_lidar_to_cam matrices in addition to the HPR offsets, so fusion models
can project LiDAR points onto each camera image without extra geometry.

Event-rate fix (v3)
-------------------
Initial NPC cooldown reduced (30–80 steps) and trigger probabilities raised
so events appear early and often within each episode.
"""

import sys, os, cv2, json, random, math
import numpy as np
from collections import deque
from panda3d.core import Vec3

sys.stdout.reconfigure(line_buffering=True)

import metadrive.engine.engine_utils as _eu
_eu._fix_offscreen_rendering = lambda: None          # Windows WGL hang fix

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.point_cloud_lidar import PointCloudLidar
from metadrive.policy.idm_policy import IDMPolicy


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

IMAGE_W, IMAGE_H   = 640, 360
LIDAR_W, LIDAR_H   = 128, 64

NUM_EPISODES               = 50
SIMULATION_FPS             = 30
EPISODE_DURATION_SECONDS   = 60        # ↓ 120→60 s — halves physics steps
MAX_STEPS                  = SIMULATION_FPS * EPISODE_DURATION_SECONDS   # 1 800

SAVE_ROOT          = "metadrive_fusion_dataset"
INTERACTION_RADIUS = 50.0           # m — enlarged from 40 so events fire sooner
MAX_CONCURRENT_EVENTS = 3

# ── Sensor sub-sampling: render sensors every N physics steps ──
# Physics runs at 30 Hz; sensors captured at 30/RENDER_EVERY_N_STEPS Hz.
# RENDER_EVERY_N_STEPS = 3  →  10 FPS effective capture rate.
# This cuts render work by ~3× with negligible impact on clip quality.
RENDER_EVERY_N_STEPS   = 3
SENSOR_FPS             = SIMULATION_FPS // RENDER_EVERY_N_STEPS   # 10

PRE_EVENT_SECONDS  = 3
POST_EVENT_SECONDS = 2
PRE_BUFFER_LEN     = PRE_EVENT_SECONDS  * SENSOR_FPS   # 30 rendered frames
POST_BUFFER_LEN    = POST_EVENT_SECONDS * SENSOR_FPS   # 20 rendered frames

# Raised probabilities → more events per episode
CUTIN_PROB_PER_SEC  = 0.05
BRAKE_PROB_PER_SEC  = 0.03

CUTIN_DURATION_RANGE = (20, 45)
BRAKE_DURATION_RANGE = (10, 25)


# ─────────────────────────────────────────────
# SURROUND-VIEW CAMERA RIG
# ─────────────────────────────────────────────
# Panda3D axes: X=right, Y=forward, Z=up.
# H=heading (yaw, +CCW from above), P=pitch, R=roll.

CAMERA_CONFIGS = {
    "front":       {"pos": (0.0,  2.0, 1.5), "hpr": (0,   -5, 0)},
    "front_left":  {"pos": (-1.5, 1.0, 1.5), "hpr": (55,  -5, 0)},
    "front_right": {"pos": ( 1.5, 1.0, 1.5), "hpr": (-55, -5, 0)},
    "back":        {"pos": (0.0, -2.0, 1.5), "hpr": (180, -5, 0)},
    "back_left":   {"pos": (-1.5,-1.0, 1.5), "hpr": (125, -5, 0)},
    "back_right":  {"pos": ( 1.5,-1.0, 1.5), "hpr": (-125,-5, 0)},
}
LIDAR_POS = (0.0, 0.0, 1.8)   # roof centre
LIDAR_HPR = (0.0, 0.0, 0.0)   # forward-facing


# ─────────────────────────────────────────────
# SHARED NPC STATE
# ─────────────────────────────────────────────

NPC_STATES = {}
EGO_REF    = [None]


# ─────────────────────────────────────────────
# RASH IDM POLICY
# ─────────────────────────────────────────────

class RashIDMPolicy(IDMPolicy):
    """IDMPolicy augmented with probabilistic cut-in and brake-check manoeuvres."""

    def __init__(self, control_object, random_seed):
        super().__init__(control_object, random_seed)
        NPC_STATES[control_object.id] = {
            "state":        "NORMAL",
            "timer":        0,
            "initial_timer":0,
            # Lowered cooldown → events fire much earlier in each episode
            "cooldown":     random.randint(30, 80),
            "direction":    0,
            "aggression":   float(np.random.uniform(0.3, 1.0)),
            "stability":    float(np.random.uniform(0.5, 1.0)),
            "event_type":   None,
        }

    def act(self, *args, **kwargs):
        idm_action = super().act(*args, **kwargs)
        vehicle = self.control_object
        state   = NPC_STATES.get(vehicle.id)
        ego     = EGO_REF[0]
        if state is None or ego is None:
            return idm_action
        s = state["state"]
        if   s == "NORMAL":   self._maybe_trigger(vehicle, ego, state); return idm_action
        elif s == "MERGING":  return self._merge_action(idm_action, state)
        elif s == "BRAKE":
            state["timer"] -= 1
            if state["timer"] <= 0:
                state.update({"state": "RECOVER", "timer": random.randint(15, 30), "event_type": None})
            return [idm_action[0], -0.8]
        elif s == "RECOVER":
            state["timer"] -= 1
            if state["timer"] <= 0:
                state.update({"state": "NORMAL", "cooldown": random.randint(60, 150), "event_type": None})
            return idm_action
        return idm_action

    def _dist(self, a, b):
        dx = a.position[0] - b.position[0]
        dy = a.position[1] - b.position[1]
        return float(np.sqrt(dx*dx + dy*dy))

    def _maybe_trigger(self, vehicle, ego, state):
        if state["cooldown"] > 0:
            state["cooldown"] -= 1
            return
        active = sum(1 for s in NPC_STATES.values() if s["state"] != "NORMAL")
        if active >= MAX_CONCURRENT_EVENTS:
            return
        if self._dist(vehicle, ego) > INTERACTION_RADIUS:
            return
        agg = state["aggression"]
        if random.random() < CUTIN_PROB_PER_SEC * agg / SIMULATION_FPS:
            dur = random.randint(*CUTIN_DURATION_RANGE)
            state.update({"state": "MERGING", "timer": dur, "initial_timer": dur,
                          "direction": random.choice([-1, 1]), "event_type": "CUT_IN"})
            return
        if self._dist(vehicle, ego) < 15:
            if random.random() < BRAKE_PROB_PER_SEC * agg / SIMULATION_FPS:
                state.update({"state": "BRAKE",
                              "timer": random.randint(*BRAKE_DURATION_RANGE),
                              "event_type": "BRAKE_CHECK"})

    def _merge_action(self, idm_action, state):
        progress  = state["timer"] / max(state["initial_timer"], 1)
        agg, stb  = state["aggression"], state["stability"]
        strength  = (0.08 + agg * 0.06) * progress
        noise     = float(np.random.normal(0, 0.01 * (1.2 - stb)))
        steering  = float(np.clip(state["direction"] * strength + noise, -1.0, 1.0))
        throttle  = float(np.clip(0.35 + agg * 0.2, -1.0, 1.0))
        state["timer"] -= 1
        if state["timer"] <= 0:
            state.update({"state": "RECOVER", "timer": random.randint(15, 30), "event_type": None})
        return [steering, throttle]


# ─────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────

def create_env():
    # map= is NUMBER OF ROAD BLOCKS (not a seed). Default 3, we use 7.
    # num_scenarios pre-allocates seed slots 0..NUM_EPISODES-1.
    return MetaDriveEnv({
        "use_render":         False,
        "image_observation":  True,
        "traffic_density":    0.30,
        "map":                7,
        "num_scenarios":      NUM_EPISODES,
        "start_seed":         0,
        "random_lane_width":  True,
        "random_lane_num":    True,
        "horizon":            MAX_STEPS,
        # "basic" spawns all vehicles at reset so assign_rash_policies sees them all.
        # Default "Trigger" keeps them in a separate queue until the ego enters each
        # block — resulting in assign_rash_policies running on an empty list.
        "traffic_mode":       "basic",
        "sensors": {
            "rgb_camera":      (RGBCamera,      IMAGE_W, IMAGE_H),
            "semantic_camera": (SemanticCamera, IMAGE_W, IMAGE_H),
            "lidar_pc":        (PointCloudLidar, LIDAR_W, LIDAR_H, True),
        },
        "vehicle_config": {"image_source": "rgb_camera"},
    })


# ─────────────────────────────────────────────
# ASSIGN RASH POLICIES — call AFTER env.reset()
# ─────────────────────────────────────────────

def assign_rash_policies(env):
    for v in env.engine.traffic_manager._traffic_vehicles:
        env.engine.add_policy(v.name, RashIDMPolicy, v, env.engine.generate_seed())


# ─────────────────────────────────────────────
# EGO LANE-KEEPING CONTROLLER
# ─────────────────────────────────────────────

def ego_controller(vehicle):
    lane = vehicle.lane
    if lane is None:
        return [0.0, 0.0]
    long, lat       = lane.local_coordinates(vehicle.position)
    lane_heading    = lane.heading_theta_at(long)
    heading_err     = vehicle.heading_theta - lane_heading
    # Normalise to (−π, π) to prevent wrap-around causing wild steering
    heading_err     = (heading_err + math.pi) % (2 * math.pi) - math.pi
    steering = float(np.clip(-lat * 0.10 - heading_err * 0.35
                             + np.random.normal(0, 0.003), -1.0, 1.0))
    return [steering, 0.35]


# ─────────────────────────────────────────────
# CALIBRATION — intrinsics + 4×4 extrinsics
# ─────────────────────────────────────────────

def _hpr_deg_to_R(hpr_deg):
    """
    Panda3D HPR (degrees, ZXY convention) → 3×3 rotation matrix.
    H = yaw around Z (heading), P = pitch around X, R = roll around Y.
    """
    H, P, R = [math.radians(x) for x in hpr_deg]
    cH, sH = math.cos(H), math.sin(H)
    cP, sP = math.cos(P), math.sin(P)
    cR, sR = math.cos(R), math.sin(R)
    RH = np.array([[ cH, -sH, 0],[ sH,  cH, 0],[  0,    0, 1]], dtype=np.float64)
    RP = np.array([[  1,   0,  0],[  0,  cP,-sP],[  0,  sP, cP]], dtype=np.float64)
    RR = np.array([[ cR,   0, sR],[  0,   1,  0],[-sR,   0, cR]], dtype=np.float64)
    return RH @ RP @ RR


def _make_T44(pos, hpr_deg):
    """4×4 sensor-to-ego transform (col-major: T @ p_sensor = p_ego)."""
    R = _hpr_deg_to_R(hpr_deg)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = pos
    return T


def build_calibration(env):
    rgb_sensor = env.engine.get_sensor("rgb_camera")
    lens = rgb_sensor.lens
    fov  = lens.getFov()
    fx   = IMAGE_W / (2.0 * math.tan(math.radians(fov[0] / 2.0)))
    fy   = IMAGE_H / (2.0 * math.tan(math.radians(fov[1] / 2.0)))
    cx, cy = IMAGE_W / 2.0, IMAGE_H / 2.0
    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    T_lidar_to_ego = _make_T44(LIDAR_POS, LIDAR_HPR)

    cal = {
        "image_width":  IMAGE_W,
        "image_height": IMAGE_H,
        "cameras":      {},
        "lidar": {
            "type":                       "PointCloudLidar",
            "channels":                   4,
            "channel_labels":             ["X_m", "Y_m", "Z_m", "intensity_norm"],
            "resolution":                 [LIDAR_W, LIDAR_H],
            "ego_centric":                True,
            "mounting_pos_vehicle_frame": list(LIDAR_POS),
            "mounting_hpr_vehicle_frame": list(LIDAR_HPR),
            "T_lidar_to_ego":             T_lidar_to_ego.tolist(),
        },
    }

    for cam_name, cfg in CAMERA_CONFIGS.items():
        T_cam_to_ego   = _make_T44(cfg["pos"], cfg["hpr"])
        # T_lidar_to_cam: project a LiDAR point into this camera's frame
        T_lidar_to_cam = (np.linalg.inv(T_cam_to_ego) @ T_lidar_to_ego).tolist()
        entry = {
            "intrinsics": {"K": K, "fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "extrinsics": {
                "pos_vehicle_frame":     list(cfg["pos"]),
                "hpr_deg_vehicle_frame": list(cfg["hpr"]),
                "T_cam_to_ego":          T_cam_to_ego.tolist(),
                "T_lidar_to_cam":        T_lidar_to_cam,
            },
        }
        cal["cameras"][f"rgb_{cam_name}"]      = entry
        cal["cameras"][f"semantic_{cam_name}"] = entry   # identical geometry

    return cal


# ─────────────────────────────────────────────
# SENSOR CAPTURE
# ─────────────────────────────────────────────

def capture_cameras(env, ego_vehicle, rgb_sensor, sem_sensor):
    node   = ego_vehicle.origin
    frames = {"rgb": {}, "semantic": {}}
    for cam_name, cfg in CAMERA_CONFIGS.items():
        pos, hpr = cfg["pos"], cfg["hpr"]
        frames["rgb"][cam_name]      = rgb_sensor.perceive(
            to_float=False, new_parent_node=node, position=pos, hpr=hpr)
        frames["semantic"][cam_name] = sem_sensor.perceive(
            to_float=False, new_parent_node=node, position=pos, hpr=hpr)
    return frames


LIDAR_HEADINGS = [0, 60, 120, 180, 240, 300]   # 6 × 60° → full 360°


def capture_lidar_360(env, ego_vehicle, lidar_sensor):
    """
    Full 360° LiDAR via six depth renders at 60° azimuth intervals.

    Bypasses MetaDrive's PointCloudLidar.get_rgb_array_cpu() (which has
    swapped cx/cy and a transposed depth map) and instead reads the raw
    float32 depth texture directly.

    Each direction's camera-frame points are rotated into ego frame via
    the standard Panda3D Z-rotation:
        R_z(H) = [[cos H, −sin H, 0],
                  [sin H,  cos H, 0],
                  [    0,      0, 1]]
    where H is the heading angle (CCW from +Y/forward, matching Panda3D).

    Intensity: 1 − dist/100  (useful range 0–100 m, not 0–far≈1000 m).

    Returns float32 array (N, 4): X, Y, Z metres (ego-centric), intensity.
    """
    cam_node    = lidar_sensor.cam
    node        = ego_vehicle.origin
    orig_parent = cam_node.getParent()
    orig_pos    = cam_node.getPos()
    orig_hpr    = cam_node.getHpr()

    # Mount camera at LiDAR position once; only heading changes per slice.
    cam_node.reparentTo(node)
    cam_node.setPos(Vec3(*LIDAR_POS))

    # Pre-compute shared intrinsics (lens params don't change between slices)
    lens       = lidar_sensor.lens
    near, far  = lens.getNear(), lens.getFar()
    fov        = lens.getFov()     # (H_fov_deg, V_fov_deg)
    fx = LIDAR_W / (2.0 * math.tan(math.radians(fov[0] / 2.0)))
    fy = LIDAR_H / (2.0 * math.tan(math.radians(fov[1] / 2.0)))
    cx = (LIDAR_W - 1) / 2.0      # correct: MetaDrive had BUFFER_H here
    cy = (LIDAR_H - 1) / 2.0      # correct: MetaDrive had BUFFER_W here
    v_idx, u_idx = np.indices((LIDAR_H, LIDAR_W), dtype=np.float32)

    all_pts = []

    for heading_deg in LIDAR_HEADINGS:
        cam_node.setHpr(Vec3(float(heading_deg), 0.0, 0.0))
        env.engine.taskMgr.step()   # commit camera transform
        env.engine.taskMgr.step()   # ensure depth buffer is flushed

        # ── Raw float32 depth [0, 1] from Panda3D depth texture ──
        depth_tex = lidar_sensor.depth_tex
        raw = np.frombuffer(depth_tex.getRamImage().getData(), dtype=np.float32)
        raw = raw.reshape(depth_tex.getYSize(), depth_tex.getXSize())  # (H, W)
        raw = raw[::-1].copy()      # OpenGL bottom-row-first → top-first

        valid_mask = raw < 0.9999   # exclude background / far-plane pixels
        z_eye = np.where(
            valid_mask,
            2.0 * near * far / ((far + near) - (2.0 * raw - 1.0) * (far - near)),
            0.0,
        ).astype(np.float32)

        # ── Back-project to camera frame (Panda3D: Y=forward, X=right, Z=up) ──
        x_cam =  (u_idx - cx) / fx * z_eye
        y_cam =  z_eye
        z_cam = -(v_idx - cy) / fy * z_eye   # v increases downward → negate

        # ── Rotate camera frame → ego frame using R_z(heading) ──
        H_rad  = math.radians(heading_deg)
        cH, sH = math.cos(H_rad), math.sin(H_rad)
        x_rot  = cH * x_cam - sH * y_cam
        y_rot  = sH * x_cam + cH * y_cam
        # z_rot = z_cam (Z-rotation leaves Z unchanged)

        # ── Add LiDAR mount offset (roof centre, same for every direction) ──
        x_ego = (x_rot + LIDAR_POS[0]).reshape(-1)
        y_ego = (y_rot + LIDAR_POS[1]).reshape(-1)
        z_ego = (z_cam + LIDAR_POS[2]).reshape(-1)

        keep = valid_mask.reshape(-1)
        x_ego, y_ego, z_ego = x_ego[keep], y_ego[keep], z_ego[keep]
        if x_ego.size == 0:
            continue

        # ── Intensity: closer → brighter, useful range 0–100 m ──
        dist      = np.sqrt(x_ego**2 + y_ego**2 + z_ego**2).astype(np.float32)
        intensity = np.clip(1.0 - dist / 100.0, 0.0, 1.0).astype(np.float32)

        all_pts.append(
            np.stack([x_ego.astype(np.float32),
                      y_ego.astype(np.float32),
                      z_ego.astype(np.float32),
                      intensity], axis=1)
        )

    # Restore camera to original state
    cam_node.reparentTo(orig_parent)
    cam_node.setPos(orig_pos)
    cam_node.setHpr(orig_hpr)

    if not all_pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(all_pts, axis=0)   # (N_total, 4)


# ─────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────

def _world_to_ego_pos(pos_world, ego_pos_world, ego_heading_rad):
    """
    Translate then rotate (−heading) to produce ego-centric XYZ.
    Heading rotation is around Z axis (yaw only).
    """
    dx = float(pos_world[0]) - float(ego_pos_world[0])
    dy = float(pos_world[1]) - float(ego_pos_world[1])
    dz = float(pos_world[2]) - float(ego_pos_world[2]) if len(pos_world) > 2 else 0.0
    cos_h = math.cos(-ego_heading_rad)
    sin_h = math.sin(-ego_heading_rad)
    x_ego =  cos_h * dx - sin_h * dy
    y_ego =  sin_h * dx + cos_h * dy
    return [x_ego, y_ego, dz]


def _wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ─────────────────────────────────────────────
# ANNOTATIONS
# ─────────────────────────────────────────────

def build_annotations(env, ego_vehicle, step, active_events):
    ego_pos     = ego_vehicle.position
    ego_heading = float(ego_vehicle.heading_theta)
    ego_pos_w   = [float(ego_pos[0]), float(ego_pos[1]), 0.0]

    nearby = []
    for obj_id, obj in env.engine.get_objects().items():
        if obj is ego_vehicle:
            continue
        if not hasattr(obj, "position") or not hasattr(obj, "speed"):
            continue

        p    = obj.position
        dx   = float(p[0]) - ego_pos_w[0]
        dy   = float(p[1]) - ego_pos_w[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > INTERACTION_RADIUS * 1.5:
            continue

        h_obj   = float(getattr(obj, "HEIGHT", 1.5))
        pos_w   = [float(p[0]), float(p[1]), h_obj / 2.0]   # centre-Z fixed
        pos_ego = _world_to_ego_pos(pos_w, ego_pos_w, ego_heading)

        heading_world = float(getattr(obj, "heading_theta", 0.0))
        heading_ego   = _wrap_angle(heading_world - ego_heading)

        vel = ([float(obj.velocity[0]), float(obj.velocity[1]), 0.0]
               if hasattr(obj, "velocity") else [0.0, 0.0, 0.0])

        obj_state  = NPC_STATES.get(getattr(obj, "id", None), {})
        is_rogue   = obj_state.get("state") in ("MERGING", "BRAKE")

        nearby.append({
            "id":    str(obj_id),
            "class": "vehicle",
            # ── World frame ──
            "position_world":  pos_w,
            "heading_rad_world": heading_world,
            # ── Ego-centric frame (use these for fusion model inputs) ──
            "position_ego":    pos_ego,
            "heading_rad_ego": heading_ego,
            # ── Box geometry ──
            "dimensions_lwh": [
                float(getattr(obj, "LENGTH", 4.5)),
                float(getattr(obj, "WIDTH",  2.0)),
                float(h_obj),
            ],
            "speed_ms":           float(obj.speed),
            "velocity":           vel,
            "distance_to_ego":    dist,
            "is_rogue":           bool(is_rogue),
            "rogue_event_type":   obj_state.get("event_type") if is_rogue else None,
        })

    return {
        "step": int(step),
        "ego": {
            "position_world": ego_pos_w,
            "heading_rad":    ego_heading,
            "speed_ms":       float(ego_vehicle.speed),
            "velocity":       ([float(ego_vehicle.velocity[0]),
                                float(ego_vehicle.velocity[1]), 0.0]
                               if hasattr(ego_vehicle, "velocity") else [0.0]*3),
        },
        "anomaly": {
            "is_event_frame": len(active_events) > 0,
            "active_events":  active_events,
        },
        "objects": nearby,
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_active_events():
    return [
        {"npc_id": vid, "type": s.get("event_type")}
        for vid, s in NPC_STATES.items()
        if s["state"] in ("MERGING", "BRAKE")
    ]


def save_event_clip(event_dir, frame_buffer):
    rgb_dir   = os.path.join(event_dir, "rgb")
    sem_dir   = os.path.join(event_dir, "semantic")
    lidar_dir = os.path.join(event_dir, "lidar")
    ann_dir   = os.path.join(event_dir, "ann")
    for d in (rgb_dir, sem_dir, lidar_dir, ann_dir):
        os.makedirs(d, exist_ok=True)

    for idx, entry in enumerate(frame_buffer):
        fid  = f"{idx:05d}"
        cams = entry["cameras"]
        for cam_name, img in cams["rgb"].items():
            cv2.imwrite(os.path.join(rgb_dir,   f"{fid}_{cam_name}.png"), img)
        for cam_name, img in cams["semantic"].items():
            cv2.imwrite(os.path.join(sem_dir,   f"{fid}_{cam_name}.png"), img)
        np.save(os.path.join(lidar_dir, f"{fid}.npy"), entry["lidar"])
        with open(os.path.join(ann_dir,  f"{fid}.json"), "w") as f:
            json.dump(entry["annotations"], f, indent=2)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

os.makedirs(SAVE_ROOT, exist_ok=True)
print("Initialising environment (first reset may take 30–60 s on Windows)…")
env = create_env()

rgb_sensor = sem_sensor = lidar_sensor = None

for episode in range(NUM_EPISODES):
    print(f"═══════════ EPISODE {episode:04d} / {NUM_EPISODES} ═══════════")

    NPC_STATES.clear()
    EGO_REF[0] = None

    env.reset(seed=episode)
    print(f"  reset OK  (seed={episode})")

    assign_rash_policies(env)

    if rgb_sensor is None:
        rgb_sensor   = env.engine.get_sensor("rgb_camera")
        sem_sensor   = env.engine.get_sensor("semantic_camera")
        lidar_sensor = env.engine.get_sensor("lidar_pc")

    if episode == 0:
        print("  Extracting calibration…")
        cal = build_calibration(env)
        with open(os.path.join(SAVE_ROOT, "calibration.json"), "w") as f:
            json.dump(cal, f, indent=2)
        print("  calibration.json saved")

    episode_dir   = os.path.join(SAVE_ROOT, f"episode_{episode:04d}")
    os.makedirs(episode_dir, exist_ok=True)

    pre_buffer     = deque(maxlen=PRE_BUFFER_LEN)
    post_countdown = 0
    post_buffer    = []
    event_counter  = 0
    event_dir      = None
    ego_terminated = False

    for step in range(MAX_STEPS):
        ego = env.agent
        if ego is not None:
            EGO_REF[0] = ego   # keep last-known reference even after termination
        ego = EGO_REF[0]

        action = ego_controller(ego) if ego is not None else [0.0, 0.0]
        _, _, terminated, truncated, _ = env.step(action)

        if terminated and not ego_terminated:
            ego_terminated = True
            print(f"  [TERM] ego terminated at step={step}")

        # ── Progress log at 10-second intervals (physics steps) ──
        if step % 300 == 0:
            spd      = ego.speed if ego is not None else 0.0
            ev_now   = get_active_events()
            print(f"  step={step:04d}  events={len(ev_now)}  "
                  f"speed={spd:.1f}m/s  clips={event_counter}  "
                  f"npcs={len(NPC_STATES)}")

        # ── Sub-sampled sensor capture (every 3rd physics step = 10 FPS) ──
        # Physics runs at 30 Hz; sensors captured at 10 FPS.
        # Skipping render steps cuts GPU work by ~3× with no impact on clips.
        if step % RENDER_EVERY_N_STEPS != 0:
            continue

        cameras = capture_cameras(env, ego, rgb_sensor, sem_sensor)
        lidar   = capture_lidar_360(env, ego, lidar_sensor)   # 360°, fixed intensity

        active_events = get_active_events()
        annotations   = build_annotations(env, ego, step, active_events)
        entry = {"cameras": cameras, "lidar": lidar, "annotations": annotations}

        # ── Event-triggered rolling-buffer save ──
        if active_events:
            if post_countdown == 0:
                event_dir   = os.path.join(episode_dir, f"event_{event_counter:04d}")
                event_counter += 1
                post_buffer = list(pre_buffer) + [entry]
            else:
                post_buffer.append(entry)
            post_countdown = POST_BUFFER_LEN
        elif post_countdown > 0:
            post_buffer.append(entry)
            post_countdown -= 1
            if post_countdown == 0:
                save_event_clip(event_dir, post_buffer)
                print(f"  [SAVED] {os.path.basename(event_dir)}  "
                      f"{len(post_buffer)} frames")
                post_buffer = []

        pre_buffer.append(entry)

    if post_buffer:
        save_event_clip(event_dir, post_buffer)
        print(f"  [FLUSH] {os.path.basename(event_dir)}  {len(post_buffer)} frames")

    print(f"  Episode {episode:04d} done — event clips: {event_counter}")

env.close()
print("Dataset collection complete.")
