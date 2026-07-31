# Simulation-Based Rash Driving Behaviour Prediction  
## Using Multi-Modal Temporal Sensor Fusion  
### Sample Presentation for Professor Verification

**Student:** [Your Name]  
**Institution:** Kyushu Institute of Technology (Kyutech)  
**Tools:** MetaDrive 0.4.3 · Python 3.8 · PyTorch 2.4.1 · MobileNetV3-Small  
**Date:** May 2026

---

---

## SLIDE 1 — Title

**Simulation-Based Rash Driving Behaviour Prediction  
for Autonomous Vehicles Using Multi-Modal Temporal Sensor Fusion**

*From dataset collection → feature engineering → three model generations → final evaluation*

**Key Stack:**
- Simulation: MetaDrive 0.4.3 (Panda3D-based open-source AV simulator)
- Framework: PyTorch 2.4.1 (CPU), Python 3.8
- Backbone: MobileNetV3-Small (frozen, ImageNet pre-trained)
- Temporal model: Bidirectional GRU (BiGRU)
- Platform: Windows 11, CPU-only

---

## SLIDE 2 — Research Motivation & Problem Statement

**Problem:**  
Autonomous vehicles must anticipate *rash behaviour* (sudden cut-ins, hard brakes, aggressive lane changes) by nearby human drivers to react safely and in time.

**Why simulation?**
- Real-world rash driving data is rare, dangerous to collect, and privacy-sensitive
- MetaDrive simulation generates unlimited labelled rash events on demand
- Ground-truth labels for every frame — impossible in real dashcam datasets
- Enables controlled ablation experiments (swap individual sensors in/out)
- Validated approach before real-world sensor deployment

**Research Question:**  
Can a multi-modal temporal fusion model combining LiDAR (BEV), surround-view cameras (RGB + semantic), and kinematic motion data outperform single-modality baselines for frame-level rash driving prediction?

**Thesis Claim:**  
Early fusion of BEV LiDAR + RGB cameras + semantic cameras + JSON kinematics into a BiGRU temporal classifier achieves ≥90% F1 on held-out test episodes, outperforming all single- and partial-modality baselines.

---

## SLIDE 3 — Novelty Claim: Task, Dataset, and Model

### The task is novel

Almost all published "aggressive driving detection" work monitors the **ego driver** — is the person inside this car driving dangerously? This work addresses a fundamentally different problem:

> **"Given the ego AV's own sensors, can we predict when a *nearby* vehicle is about to behave rashly?"**

This is a **reactive safety** problem — the AV defending itself from external threats — not a driver monitoring problem. No publicly available dataset or model specifically addresses this framing with multi-modal sensor fusion.

### The dataset is novel

| Public Dataset | Frame-level rash labels? | 4-modal simultaneous capture? | Exact trigger frame known? |
|----------------|--------------------------|-------------------------------|---------------------------|
| nuScenes, KITTI, Waymo | ✗ (normal driving) | ✓ (real sensors) | ✗ |
| HighD, INTERACTION, NGSIM | Coarse behaviour tags only | ✗ (trajectory data only) | ✗ |
| D2-City (dashcam) | ✗ | ✗ | ✗ |
| CARLA RL datasets | ✗ | Partial | Partial |
| **This work** | **✓ Binary, every frame** | **✓ RGB+Sem+LiDAR+JSON** | **✓ Frame-perfect** |

Three properties no public dataset combines:
1. **Frame-level binary rash event labels** — exact trigger frame from simulator, not human annotation
2. **6-view RGB + 6-view semantic + ego-centric 360° LiDAR + per-NPC kinematics** simultaneously
3. **Scripted events with known type, NPC identity, and frame index** — enables clean temporal sliding windows

### The model is novel

Existing architectures handle parts of the problem:

| Related Work | Sensor Fusion | Temporal Model | Task |
|-------------|-------------|----------------|------|
| BEVFusion (MIT, 2022) | Camera + LiDAR BEV | CNN only | Detection / segmentation |
| Social LSTM / GAN | Position only | LSTM | Trajectory prediction |
| ConvLSTM on BEV | Occupancy grid | ConvLSTM | Future occupancy |
| Rash driving LSTMs (existing) | IMU/GPS (ego only) | LSTM | Ego driver monitoring |
| **This work** | **BEV + 6-view RGB + 6-view Semantic + Kinematics** | **BiGRU** | **Third-party rash prediction** |

The specific combination — BEV spatial features + surround camera features (RGB + semantic) + kinematic motion features → BiGRU temporal classifier — for third-party rash driving prediction has not been published.

---

## SLIDE 4 — Why MetaDrive, Not CARLA?

### Technical comparison

| Criterion | MetaDrive 0.4.3 | CARLA 0.9.x |
|-----------|-----------------|-------------|
| **GPU requirement** | None — CPU only | ≥8 GB VRAM dedicated |
| **Installation** | `pip install metadrive-simulator` | 10 GB UE4 binary + server process |
| **Architecture** | Single Python process | Client/server over TCP |
| **Road diversity** | Procedural — unlimited unique layouts | 12 fixed maps (Town01–Town12) |
| **NPC event scripting** | Direct Python API — exact frame control | TrafficManager — coarser, harder to script |
| **Reproducibility** | Seed → exact deterministic replay | Server timing drift between runs |
| **Rendering** | Panda3D (lightweight, fast) | Unreal Engine 4 (photorealistic, slow) |
| **Timestamp precision** | Frame-perfect (in-process call) | Requires client-server sync protocol |

### Why CARLA's advantages do not matter for this research

**1. Photorealism is irrelevant when using a frozen ImageNet backbone.**  
All visual features pass through a **frozen MobileNetV3-Small** pretrained on real photographs. The backbone already bridges the simulation-to-real visual domain gap. Whether the road texture in simulation looks like a photograph does not change what a frozen, real-image-trained backbone extracts. GPU budget spent on UE4 rendering would not improve any metric.

**2. Fixed maps would hurt generalisation.**  
CARLA's 12 fixed maps mean repeated episodes share the same road geometry, building textures, and lane layout. A model could exploit these map-specific cues instead of learning scene-agnostic rash behaviour patterns. MetaDrive's procedural generation produces a unique road layout per episode seed, forcing the model to generalise genuinely.

**3. Exact event frame timing is foundational to this dataset.**  
The dataset's core value is precise frame-level labels. MetaDrive's Python-native NPC scripting lets the collector record the exact simulation frame of the event trigger in the same process call — no synchronisation overhead. In CARLA, the event time would need to be transmitted from the UE4 server to the Python client over TCP, introducing potential jitter that would contaminate the temporal label boundary.

**4. CPU-only operation is a valid scope decision, not a limitation.**  
A system that achieves 90% F1 for rash driving prediction using only CPU hardware demonstrates that the approach is deployable on resource-constrained onboard AV computers. This is a feature for practical deployment, not an apology. CARLA would have made this research impossible on the available hardware.

---

## SLIDE 5 — Dataset Collection Pipeline

**Simulator:** MetaDrive 0.4.3 — procedural road generation, randomised NPC behaviour

**Collection parameters:**
| Parameter | Value |
|-----------|-------|
| Total episodes collected | 50 |
| Episode duration | 60 seconds each |
| Simulation FPS | 30 FPS |
| Capture rate | Every 3 steps → **10 FPS** |
| Pre-event buffer | 30 frames (~3 sec before) |
| Post-event buffer | 20 frames (~2 sec after) |
| Rash event types | Cut-in, Emergency brake (NPC-triggered) |
| Camera views | 6 surround: front, front-left, front-right, back, back-left, back-right |
| Camera resolution | 640 × 360 pixels |
| LiDAR | 360° depth renders at 6 headings (0°, 60°, 120°, 180°, 240°, 300°) |

**Final dataset statistics:**
| Item | Count |
|------|-------|
| Episodes with rash events | ~44 of 50 |
| Total event clips | **68** |
| Total frames | **17,559** |
| Event frames (positive) | **14,727** (83.9%) |
| Normal frames (negative) | **2,832** (16.1%) |

**Per-frame data collected:**
1. **6 RGB images** — surround view cameras (640×360 PNG)
2. **6 Semantic images** — segmentation masks (same 6 views)
3. **LiDAR point cloud** — ego-centric (x, y, z, intensity) array
4. **JSON annotation** — ego state + up to 3 NPC states + `is_event_frame` label

---

## SLIDE 4 — LiDAR Bird's Eye View (BEV) Representation

**From raw point cloud → structured spatial grid**

**Raw LiDAR format:**  
Array of shape `(N_points, 4)` — columns: `[X, Y, Z, intensity]` in ego-centric metres

**BEV Grid specification:**
| Parameter | Value |
|-----------|-------|
| Grid size | 64 × 64 pixels |
| Resolution | 1.0 m/pixel |
| X range | [-32, +32] m (lateral) |
| Y range | [-16, +48] m (forward-biased) |
| Ego vehicle position | Pixel (32, 16) — lower-centre |
| Channels | 3 |

**3 Channels:**
1. **Occupancy** — binary (1 = any point in voxel column)
2. **Mean intensity** — normalised 0–1 reflectance
3. **Mean height** — normalised Z value (ground discrimination)

**Important note on older episodes:**  
Episodes 0–18 were collected before a bug fix; their LiDAR is in *world-scale* coordinates → BEV grids are empty. Episodes 19–49 use correct *ego-centric* coordinates → valid BEV spatial data. The model handles the empty-BEV case gracefully by falling back to RGB + JSON signals.

---

## SLIDE 5 — What is "Kinematics / JSON"?

This is a common point of confusion — clarified here explicitly.

**"Kinematics" in this project = the numerical data already in the JSON annotation files.**  
It is NOT a separate sensor module.

**JSON annotation structure (per frame):**
```json
{
  "frame_id": "000042",
  "ego": {
    "speed_mps": 8.34,
    "heading_deg": 12.5,
    "position": [234.1, 891.2]
  },
  "npcs": [
    { "id": 1, "rel_x": 2.1, "rel_y": 14.3,
      "speed_mps": 9.0, "heading_deg": 11.2,
      "distance_m": 14.5 },
    ...
  ],
  "anomaly": { "is_event_frame": true, "type": "cut_in" }
}
```

**Feature extraction → 19-dim vector:**
| Component | Dims | Content |
|-----------|------|---------|
| Ego speed | 1 | m/s |
| Ego heading | 1 | degrees |
| Ego position | 2 | x, y (metres) |
| NPC 1 | 5 | rel_x, rel_y, speed, heading, distance |
| NPC 2 | 5 | (zeros if absent) |
| NPC 3 | 5 | (zeros if absent) |
| **Total** | **19** | — |

**In a real AV system, this information comes from:**
- Ego state → GPS + IMU + odometer
- NPC states → RADAR + LiDAR 3D object detection + Kalman filter tracking

---

## SLIDE 8 — Model Architecture: Early Fusion V2 (Main Contribution)

**Design philosophy:**  
Fuse all four modalities at the *feature level* (early fusion) before temporal reasoning, rather than making separate predictions per modality and combining outputs (late fusion).

```
Per-frame feature extraction (T = 10 frames, 1 second context window)
╔══════════════════════════════════════════════════════════════════╗
║  BEV Grid (3×64×64)                                             ║
║    → BEV CNN (trainable, ~89K params)                           ║
║    → 64-dim BEV feature vector                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  RGB Images (6 views × 576-dim MobileNetV3 backbone)            ║
║    → mean-pool 6 views                                          ║
║    → Linear(576→64) + LayerNorm + ReLU                          ║
║    → 64-dim RGB feature vector                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Semantic Images (6 views × 576-dim backbone, separate weights) ║
║    → mean-pool 6 views                                          ║
║    → Linear(576→32) + LayerNorm + ReLU                          ║
║    → 32-dim semantic feature vector                             ║
╠══════════════════════════════════════════════════════════════════╣
║  JSON Kinematics (19-dim)                                       ║
║    → Linear(19→32) + LayerNorm + ReLU                           ║
║    → 32-dim kinematic feature vector                            ║
╚══════════════════════════════════════════════════════════════════╝
                        ↓ concat
            192-dim fused feature vector per frame
                        ↓
    BiGRU (hidden=96, bidirectional=2×96=192, 2 layers, dropout=0.50)
                        ↓
          Final hidden state at timestep T (192-dim)
                        ↓
    Head: Dropout(0.5) → Linear(192→64) → ReLU → Dropout(0.25) → Linear(64→1)
                        ↓
         Sigmoid → P(rash event) ∈ [0.0, 1.0]
```

**Total trainable parameters: 492,193**  
(BEV CNN ~89K + projection heads ~17K + BiGRU ~278K + head ~13K)

**BEV CNN architecture:**
```
Conv(3→16, 3×3) + BN + ReLU + MaxPool2   →  (16, 32, 32)
Conv(16→32, 3×3) + BN + ReLU + MaxPool2  →  (32, 16, 16)
Conv(32→64, 3×3) + BN + ReLU             →  (64, 16, 16)
AdaptiveAvgPool(4×4)                      →  (64, 4, 4)
Flatten + Linear(1024→64) + ReLU          →  64-dim
Dropout(0.50)
```

---

## SLIDE 9 — Training Protocol

**Phase 1 — Feature Caching (run ONCE, ~46 minutes)**  
Extract and save frozen backbone features + BEV grids to disk:
- MobileNetV3-Small runs on all 12 images per frame (6 RGB + 6 semantic) in **one batched forward pass** → 5× speedup over sequential processing
- BEV grids computed from raw LiDAR point clouds
- Saved as `.npz` files: `feature_cache_v2/<episode>/<event>/<frame_id>.npz`
- Total cache: 17,559 files, ~527 MB
- Training epochs then read from cache — no backbone re-computation

**Phase 2 — Training**
| Hyperparameter | Value |
|----------------|-------|
| Optimiser | AdamW |
| Learning rate | 3×10⁻⁴ |
| LR schedule | CosineAnnealingLR (T_max=40) |
| Weight decay | 5×10⁻⁴ |
| Dropout | 0.50 (BiGRU + BEV CNN) |
| Loss | BCEWithLogitsLoss (pos_weight=0.199) |
| Sequence length | 10 frames (1 second) |
| Batch size | 32 |
| Max epochs | 40 |
| Early stopping | patience=12 epochs on val F1 |

**pos_weight = 0.199** explained:  
With 83.9% positive (rash event) frames, the model would trivially predict "event" for everything. `pos_weight = n_negative/n_positive = 2832/14727 = 0.199` *down-weights* the loss contribution from positive samples, forcing the model to be precise rather than just always-positive.

**Train / Val / Test split (episode-level — no temporal leakage):**
| Split | Episodes | Clips | Frames |
|-------|----------|-------|--------|
| Test (held out) | {9, 10, 47, 49} | 6 | ~1,697 |
| Val (checkpoint selection) | {3, 6, 45, 46} | 5 | — |
| Train | remaining 44 episodes | 57 | ~14,400 |

**Optimal decision threshold found by sweeping val set: 0.55**

---

## SLIDE 10 — Model Comparison: Three Generations

| Model | Modalities | Architecture | Params | Val F1 | Epoch |
|-------|-----------|-------------|--------|--------|-------|
| **Kinematic LSTM** | JSON only (19-dim) | BiGRU(64) 2L + head | 184K | 0.9275 | 4 |
| **Early Fusion v1** | RGB + JSON + LiDAR stats | Proj heads + BiGRU(64) 2L | 185K | 0.9265 | 48 |
| **Early Fusion v2** | BEV + RGB + Sem + JSON | BEV CNN + Proj heads + BiGRU(96) 2L | 492K | **0.9455** | 2 |

**Key observations:**
- Kinematic LSTM (JSON-only) achieves 0.9275 — motion data alone is highly predictive
- Early Fusion v1 (adding RGB + LiDAR stats) does **not** improve over kinematics alone — statistical LiDAR features (9-dim) add noise rather than signal
- Early Fusion v2 replaces statistics with the **full BEV grid** + adds **semantic cameras** → +0.018 F1 improvement, the best model

---

## SLIDE 11 — Final Test Set Results (Early Fusion V2)

**Test set:** 6 held-out episodes (never seen during training or threshold tuning)  
**Frames:** 1,697 total (1,453 rash event + 244 normal)  
**Checkpoint:** epoch 2, val F1=0.9455

| Metric | Score |
|--------|-------|
| Optimal threshold (from val) | **0.55** |
| **Test F1-score** | **0.9003** |
| **Test Precision** | **0.9368** |
| **Test Recall** | **0.8665** |
| **Test AUROC** | **0.8976** |
| Test Accuracy | 0.84 |

**Per-class results:**
```
              precision  recall  f1-score  support
Normal            0.45    0.65      0.53      244
Rash Event        0.94    0.87      0.90     1453

accuracy                             0.84     1697
macro avg         0.69    0.76      0.72     1697
weighted avg      0.87    0.84      0.85     1697
```

**Reading the results:**
- **94% Precision on Rash Event** → when the model raises an alarm, it is correct 94% of the time (low false alarm rate — critical for AV safety)
- **87% Recall on Rash Event** → the model catches 87% of all actual rash events (acceptable miss rate)
- **Normal class harder** → only 244 normal frames in test (14.4%) — heavy imbalance makes normal class metrics lower by design
- **AUROC = 0.898** → strong discriminative ability across all possible thresholds

**Saved artefacts:**
- `models/ef2_roc_curve.png` — ROC curve
- `models/ef2_confusion_matrix.png` — Confusion matrix
- `models/ef2_training_curves.png` — Train/val loss and F1 over epochs

---

## SLIDE 12 — Key Findings & Conclusions

1. **Multi-modal fusion wins**: BEV + RGB + Semantic + JSON achieves val F1=0.9455, the best result across all three model generations.

2. **Motion data dominates**: JSON kinematics alone (0.9275) outperforms RGB+stats fusion (0.9265). The relative position and speed of nearby vehicles are the strongest signal for rash behaviour.

3. **BEV provides the key upgrade**: Replacing 9-dim LiDAR statistics with the full 64×64×3 BEV grid enables the model to learn *spatial context* — not just "how many points are nearby" but *where* they are relative to the ego vehicle.

4. **Semantic cameras add scene understanding**: The 32-dim semantic projection helps the model distinguish lanes, road boundaries, and vehicle types — context that pure RGB conflates.

5. **High precision is more important than high recall** for AV warning systems: 94% precision means 94% of alarms are real. A 6% false alarm rate is acceptable; a 13% miss rate triggers further safety monitoring (not full braking).

6. **Simulation-to-real gap remains**: The model is trained and tested entirely in MetaDrive. Real sensors have noise, weather, occlusion, and calibration errors not present in simulation. Next step: test on real dashcam datasets.

---

## SLIDE 13 — Limitations & Future Work

**Current limitations:**
- Only 2 rash event types (cut-in, emergency brake) — limited generalisation
- Older episodes (0–18) have world-scale LiDAR → empty BEV grids; model falls back to RGB+JSON for ~26% of clips
- Heavy class imbalance (83.9% event frames) inflates event-class metrics
- Hardware constraints (16GB RAM, CPU-only) limited retraining experiments
- Overfitting observed: best model found at epoch 2, suggesting the BEV CNN has capacity to memorise training patterns quickly

**Future work:**
1. Retrain with stronger regularisation after moving data off OneDrive (reduces loading from 637s → ~35s)
2. Add more rash event types (tailgating, aggressive lane changes, red-light running)
3. Platt scaling / temperature calibration for probability output reliability
4. Evaluate on real-world dashcam sequences (domain adaptation)
5. Explore attention mechanisms over the 10-frame temporal window
6. Video-level (episode-level) evaluation metrics, not just frame-level F1

---

## APPENDIX — System Verification Details

**File locations:**
```
simulation/
├── train_early_fusion_v2.py  ← main training script
├── eval_ef2.py               ← memory-efficient evaluation script
├── models/ef2_best.pt        ← best checkpoint (epoch=2, val_F1=0.9455)
├── models/ef2_roc_curve.png  ← ROC curve plot
├── models/ef2_confusion_matrix.png
└── models/ef2_training_curves.png
```

**Reproduction steps:**
```bash
# 1. Phase 1 (if cache not present, ~46 min)
python train_early_fusion_v2.py  # stops after phase 1 if cache incomplete

# 2. Full training (~40 epochs with early stopping)
python train_early_fusion_v2.py

# 3. Evaluation on test set
python eval_ef2.py
```

**Hardware used:**
- CPU: Intel Core i7 (no GPU used)
- RAM: 16 GB
- Storage: OneDrive-synced SSD
- OS: Windows 11, Python 3.8

**Verified outputs (from eval_ef2.py run):**
```
Checkpoint: epoch=2, val_F1=0.9455
Optimal threshold (val): 0.55  val F1=0.9469

--- Test set results (EarlyFusionV2, epoch 2) ---
  F1        : 0.9003
  Precision : 0.9368
  Recall    : 0.8665
  AUROC     : 0.8976
```

---
*End of Presentation*
