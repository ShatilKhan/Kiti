# Camera Motion Compensation - Research Context & Reference Guide

**Created:** 2026-02-15
**Purpose:** Learning reference for camera motion compensation techniques relevant to the Kiti autonomous vehicle obstacle recognition pipeline.

---

## Table of Contents
1. [Overview](#overview)
2. [Paper Summaries](#paper-summaries)
3. [Key Concepts & Techniques](#key-concepts--techniques)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Relevance to Kiti Project](#relevance-to-kiti-project)
6. [Further Reading & References](#further-reading--references)

---

## Overview

Camera Motion Compensation (CMC) is the process of separating **ego-motion** (the camera's own movement) from **independent object motion** in video sequences. This is fundamental for autonomous vehicles where the camera is always moving with the vehicle, yet we need to detect obstacles that are moving independently in the scene.

### Why It Matters for Autonomous Driving
When a vehicle-mounted camera moves, *everything* in the image appears to move. The challenge is distinguishing:
- **Background motion** caused by the vehicle/camera moving (ego-motion)
- **Foreground motion** caused by actual moving obstacles (pedestrians, other vehicles)

Without CMC, background subtraction and motion detection methods fail because they interpret camera-induced motion as object motion.

---

## Paper Summaries

### Paper 1: Computational Complexity of Motion Estimation in MPEG-4 Encoder
**Author:** Muhammad Shahid | **Year:** 2010 | **Type:** MSc Thesis, Blekinge Institute of Technology

**What it's about:** Reducing the computational cost of block-matching motion estimation for video compression on mobile devices.

**Key Ideas:**
- **Full Search (FS)** checks every possible block position - optimal but extremely slow
- **SLIMPEG** is already 99% more efficient than FS
- Four proposed algorithms exploit spatial/temporal correlation:
  1. **Spatial Correlation:** Neighboring blocks in the same frame have similar motion vectors. Use average of neighbors as estimate.
  2. **Temporal Correlation:** Same block position across consecutive frames has similar motion. Reuse previous frame's MV.
  3. **Adaptive SAD Control:** Dynamically trade off quality vs. speed using a threshold on the Sum of Absolute Differences.
  4. **Combined:** Merge spatial + temporal approaches.

**Core Metric - SAD (Sum of Absolute Differences):**
```
SAD = Σᵢ Σⱼ |Current(i,j) - Reference(i,j)|
```
This measures how "different" two blocks are. Lower SAD = better match.

**Takeaway for Kiti:** SAD-based block matching is foundational for stereo vision and object tracking. Understanding computational shortcuts (spatial/temporal correlation) helps when optimizing real-time processing.

---

### Paper 2: Compensation Method for Rotary-Scan Space Camera
**Authors:** Yingying Sun et al. | **Year:** 2020 | **Journal:** Optical and Quantum Electronics

**What it's about:** Compensating for image blur caused by a rotating satellite camera. The camera rotation introduces >200 pixels of image motion.

**Key Ideas:**
- Image motion is decomposed into 4 physical sources:
  - V₁: Satellite orbital flight
  - V₂: Camera rotation
  - V₃: Earth rotation
  - V₄: Satellite attitude variation (pitch, yaw, roll)
- **Total motion:** V = V₁ + V₂ + V₃ + V₄
- Two-step compensation:
  1. **Optical:** Fast Steering Mirror (FSM) compensates large tangential + radial motion
  2. **Mechanical:** Piezoelectric transducer rotates detector to handle residual motion
- Result: Motion reduced from 220 pixels to 4 pixels (98% compensation)

**Takeaway for Kiti:** The principle of **decomposing total image motion into separate physical components** is directly analogous to decomposing vehicle ego-motion into translational and rotational components. This decomposition approach is key to building robust CMC for vehicle cameras.

---

### Paper 3: Feature Tracking and Motion Compensation for Action Recognition ⭐ HIGH RELEVANCE
**Authors:** H. Uemura, S. Ishikawa, K. Mikolajczyk | **Year:** 2008 | **Venue:** BMVC

**What it's about:** Separating camera motion from human action motion in realistic video sequences to enable robust action recognition.

**Key Pipeline:**
1. **Feature Detection:** Multiple detectors (MSER, Harris-Laplace, Hessian-Laplace, FAST) + grid points → ~3000 features/frame
2. **Feature Tracking:**
   - KLT tracker (pyramidal, 4 levels, 15×15 patches)
   - SIFT descriptor matching with Bhattacharyya distance
3. **Motion Compensation via Dominant Plane:**
   - Mean Shift segmentation → identify image regions
   - RANSAC per segment → estimate homographies
   - Build Feature-to-Segment matrix: `S_f(Sm, fi) = 1` if feature fi ∈ segment Sm
   - Build Segment-Homography matrix: `H_S = S_f × H_f`
   - Select dominant homography: column with max sum in H_S
   - Merge segments with >80% inliers
   - Up to 3 dominant planes per frame
4. **Subtract dominant motion** to isolate independent object motion

**Results:**
- Background motion reduced by 80%
- Foreground motion only reduced by 30% (preserved!)
- Foreground/background discriminability improved 4× (ratio from 1.05 → 4.1)

**Takeaway for Kiti:** This is the closest to what Kiti needs. The approach of estimating a **dominant plane homography** via RANSAC and subtracting it from all tracked features is a proven method for separating ego-motion from obstacle motion. The 4× improvement in foreground/background discrimination shows this technique's power.

---

### Paper 4: Synthetic Camera Poses for Soccer Video Calibration
**Authors:** Mavrogiannis & Maglogiannis | **Year:** 2024 | **Journal:** Multimedia Tools and Applications

**What it's about:** Calibrating multiple camera positions in soccer stadiums using synthetic training data and deep learning.

**Key Ideas:**
- **Two-stage approach:**
  1. Camera Localization (one-time): EfficientNet-B0 regresses 3D camera coordinates
  2. Per-frame Parameter Estimation: EfficientNet-B0 regresses pan, tilt, focal length
- **ECC Algorithm** (Enhanced Correlation Coefficient) for homography refinement
- **Synthetic data generation** to train calibration models

**Key Equations:**
- Tilt: `φ = arctan(z/b)` where z = camera elevation, b = w + y
- Pan: `θ = arctan(a/b)` where a = distance from reference
- Focal length: `F = P × (D/H)` where P = pixel size, D = distance, H = height

**Takeaway for Kiti:** Camera calibration (intrinsic/extrinsic parameters) and homography estimation are prerequisites for accurate distance estimation and ground-plane projection in Kiti's pipeline. The ECC refinement technique could improve lane detection and road plane estimation.

---

### Paper 5: Moving Object Detection for Moving Camera ⭐ HIGH RELEVANCE
**Authors:** Yang Yu, L. Kurnianggoro, K.-H. Jo | **Year:** 2019 | **Journal:** IJCAS

**What it's about:** Real-time detection of independently moving objects from a moving camera using homography-based compensation and adaptive background modeling.

**Key Pipeline:**
1. **Grid-based key point tracking:** 16×16 grid, pyramidal Lucas-Kanade optical flow
2. **Homography estimation via RANSAC:** `x_t = H_{t-1} × x_{t-1}`
3. **Frame alignment:** Warp previous frame to current using homography
4. **Adaptive background model** with two models:
   - Background model M (for subtraction)
   - Candidate model C (for stability check)
   - Candidate age mechanism with **adaptive threshold** that decays exponentially with optical flow speed
5. **Background subtraction** with local pixel difference and consistency check
6. **Post-processing:** Gaussian filter, connected components, erosion, dilation

**Key Equations:**
- Background model update: `M_t(x) = (1-γ) × M^T_{t-1}(x) + γ × I_t(x)`
- Adaptive threshold: `τ_α = μ × e^{-σ × f̄}` (exponential decay with average optical flow magnitude)
- This means: **faster camera motion → lower age threshold → faster background adaptation**

**Results:** 14.8-51.2 FPS depending on resolution. F1=0.569 on ChangeDetection PTZ dataset.

**Takeaway for Kiti:** This paper directly addresses Kiti's core challenge - detecting moving objects from a moving camera. The homography-based alignment + adaptive background model is a practical, real-time approach. The adaptive threshold that responds to camera speed is particularly clever for vehicle scenarios where speed varies constantly.

---

### Paper 6: Real-Time Camera Ego-Motion Compensation on GPU ⭐ HIGH RELEVANCE
**Authors:** Johan Hedborg, Bjorn Johansson | **Year:** ~2008 | **Institution:** Linköping University (IVSS project)

**What it's about:** Simultaneous ego-motion compensation and lens undistortion for traffic intersection cameras, running in real-time on GPU.

**Key Ideas:**
- KLT tracker on regularly sampled grid (NOT Harris corners - avoids being "stolen" by high-contrast moving vehicles)
- Two overlapping grids that alternate reset
- **5-parameter ego-motion model:** 3D translation T + 3D rotation Ω
- FOV lens distortion model: `r_d = arctan(r_u × ω) / ω`
- **GPU implementation** (DirectX): 30 FPS total pipeline

**Results:**
- GPU KLT: 8.7ms for 256 patches vs CPU reference
- Lens undistortion: GPU 0.45ms vs CPU 115ms (250× speedup)
- Background subtraction improved significantly with ego-motion compensation

**Takeaway for Kiti:** Demonstrates that simultaneous ego-motion compensation + lens correction is feasible in real-time. The insight about using **regular grid points** instead of Harris corners (to avoid tracking moving objects) is important for Kiti's pipeline design. GPU acceleration is essential for real-time AV processing.

---

### Paper 7: Image Compensation for SDLT-1 Satellite
**Authors:** Jie Huang, Juntong Xi | **Year:** 2022 | **Journal:** J. Indian Soc. Remote Sensing

**What it's about:** Compensating image motion for a geostationary satellite with a large-format (10,240×10,240) detector, handling multiple error sources.

**Key Ideas:** Systematic error modeling through coordinate transformation chains (camera → satellite → Earth-centered → Earth surface). 12-factor calibration for micro-vibration and heat flux errors.

**Takeaway for Kiti:** Low direct relevance (satellite domain), but the systematic approach to multi-source error decomposition is a useful conceptual framework.

---

### Paper 8: UCMCTrack - Multi-Object Tracking with Uniform CMC ⭐ HIGHEST RELEVANCE
**Authors:** Kefu Yi et al. | **Year:** 2024 | **Venue:** AAAI-24

**What it's about:** Multi-object tracking that is robust to camera motion, using ground-plane projection and a novel distance metric instead of expensive per-frame CMC.

**Key Innovations:**
1. **Uniform Camera Motion Compensation (UCMC):** Apply the SAME compensation parameters across an entire video sequence instead of per-frame. Dramatically reduces computation.
2. **Ground Plane Motion Modeling:** Project detections from image plane → ground plane using camera intrinsic/extrinsic parameters, then apply Kalman filter with Constant Velocity model ON THE GROUND PLANE.
3. **Mapped Mahalanobis Distance (MMD):** Replaces IoU as the distance metric. Handles cases where bounding boxes don't overlap at all.
4. **Correlated Measurement Distribution (CMD):** Maps image-plane detection noise to correlated distributions on the ground plane.
5. **Process Noise Compensation (PNC):** Models camera-motion-induced acceleration as process noise in Kalman filter.

**Key Equations:**
- Image-to-ground projection: `[u,v,1]ᵀ = A × (1/γ) × [x,y,1]ᵀ`
- Mapped measurement noise: `R_k = C × R^{uv}_k × Cᵀ`
- Mahalanobis Distance: `D = εᵀ × S⁻¹ × ε + ln|S|`

**Results:**
- MOT17: HOTA 65.8 (state-of-the-art)
- DanceTrack: HOTA 63.6 (+2.3 over prior SOTA)
- **KITTI (autonomous driving):** Car HOTA 77.1, Pedestrian HOTA 55.2
- **Speed:** >1000 FPS for association (CPU only!)

**Takeaway for Kiti:** This is the most directly applicable paper. The key insight is that tracking on the **ground plane** (not image plane) makes predictions physically meaningful - vehicles and pedestrians move on the ground, not in pixel space. The >1000 FPS speed makes it viable for real-time AV systems. Kiti should consider adopting this approach for its tracking pipeline.

---

## Key Concepts & Techniques

### 1. Homography Estimation
A **homography** is a 3×3 transformation matrix that maps points from one image to another, assuming the scene is a plane (or the camera only rotates). It's the most common tool for CMC.

**How it works:**
1. Track feature points between frames (KLT, SIFT, etc.)
2. Use RANSAC to robustly estimate the 3×3 homography matrix H
3. Warp the previous frame using H to align with the current frame
4. Anything that doesn't align = independently moving object

**Limitation:** Assumes a dominant plane (usually the ground). Fails with multiple dominant planes or strong parallax.

### 2. Optical Flow for Motion Analysis
- **Dense Optical Flow (Farneback):** Computes flow for every pixel. Already used in Kiti.
- **Sparse Optical Flow (Lucas-Kanade/KLT):** Computes flow for selected feature points. Faster, used for CMC.
- **FlowNet2.0:** Deep learning-based optical flow (used in Paper: Efficient Optical Flow Motion Detection)

### 3. Background Subtraction with CMC
Traditional background subtraction (like MOG2 in Kiti) assumes a static camera. With CMC:
1. Align consecutive frames using homography
2. Transfer the background model using the same transformation
3. Apply background subtraction on the aligned frames

### 4. Ground Plane Projection
Instead of tracking in image coordinates (pixels), project detections onto the ground plane using camera calibration parameters. This makes motion models physically meaningful.

### 5. Kalman Filter for Tracking
Used in both Kiti (current) and UCMCTrack for predicting object trajectories. Key difference: applying it on the ground plane vs. image plane.

---

## Mathematical Foundations

### Homography Transformation
```
x' = H × x

where H = | h11 h12 h13 |
          | h21 h22 h23 |
          | h31 h32 h33 |
```

### Sum of Absolute Differences (SAD)
```
SAD(dx,dy) = Σᵢ Σⱼ |I_current(i,j) - I_reference(i+dx, j+dy)|
```

### PSNR (Peak Signal-to-Noise Ratio)
```
PSNR = 10 × log₁₀[(2ⁿ - 1)² / MSE] dB
```

### Adaptive Background Model Update
```
M_t(x) = (1 - γ) × M_{t-1}^T(x) + γ × I_t(x)
```
Where γ is the learning rate and M_{t-1}^T is the previous model warped by the homography.

### Mahalanobis Distance (for tracking)
```
D = εᵀ × S⁻¹ × ε + ln|S|
```
Where ε is the residual and S is the residual covariance. This accounts for uncertainty in measurements.

---

## Relevance to Kiti Project

### Current Kiti State
- Uses MOG2 background subtraction (assumes static camera)
- Uses dense optical flow (Farneback) for motion detection
- Uses SORT tracker with Kalman filter on image plane
- Uses focal-length-based distance estimation

### Recommended Improvements Based on Papers

| Priority | Improvement | Based On | Effort |
|----------|------------|----------|--------|
| 1 | Add homography-based frame alignment before background subtraction | Papers 3, 5 | Medium |
| 2 | Move Kalman filter tracking to ground plane | Paper 8 (UCMCTrack) | Medium |
| 3 | Replace IoU with Mapped Mahalanobis Distance in SORT | Paper 8 | Medium |
| 4 | Use regular grid KLT for ego-motion estimation (not Harris corners) | Paper 6 | Low |
| 5 | Add adaptive background model threshold based on vehicle speed | Paper 5 | Low |
| 6 | Implement Uniform CMC (constant params per sequence) | Paper 8 | Low |
| 7 | Add lens distortion correction | Paper 6 | Medium |

---

## Further Reading & References

### Foundational Papers (cited across multiple papers)
- **Lucas-Kanade Optical Flow:** Lucas, B.D. & Kanade, T. (1981). "An iterative image registration technique with an application to stereo vision."
- **RANSAC:** Fischler, M.A. & Bolles, R.C. (1981). "Random sample consensus: a paradigm for model fitting."
- **KLT Tracker:** Tomasi, C. & Kanade, T. (1991). "Detection and tracking of point features."
- **MOG2 Background Subtraction:** Zivkovic, Z. (2004). "Improved adaptive Gaussian mixture model for background subtraction."
- **SORT Tracker:** Bewley, A. et al. (2016). "Simple online and realtime tracking."

### Web Resources
- OpenCV Homography Tutorial: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- OpenCV Optical Flow: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- UCMCTrack GitHub: https://github.com/corfyi/UCMCTrack
- FlowNet2.0: https://github.com/NVIDIA/flownet2-pytorch

### Key Datasets for Testing
- **KITTI** (autonomous driving): http://www.cvlibs.net/datasets/kitti/
- **MOT17/MOT20** (multi-object tracking): https://motchallenge.net/
- **DAVIS2016** (video object segmentation): https://davischallenge.org/
- **ChangeDetection** (moving camera): http://changedetection.net/
