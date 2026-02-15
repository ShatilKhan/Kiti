# Dynamic Motion Estimation - Research Context & Reference Guide

**Created:** 2026-02-15
**Purpose:** Learning reference for dynamic motion estimation techniques relevant to the Kiti autonomous vehicle obstacle recognition pipeline.

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

Dynamic Motion Estimation (ME) is the process of computing how objects or regions move between video frames. In video coding, ME is used for compression. In autonomous driving, the same mathematical foundations power:
- **Object tracking** (where did this car move between frames?)
- **Ego-motion estimation** (how did our vehicle move?)
- **Obstacle trajectory prediction** (where will this pedestrian be in 2 seconds?)

This collection covers motion estimation from multiple angles: software optimization, hardware acceleration, energy efficiency, and novel search strategies.

---

## Paper Summaries

### Paper 1: Fast Affine Motion Estimation for VVC ⭐ HIGH RELEVANCE
**Authors:** Sang-hyo Park, Je-Won Kang | **Year:** 2019 | **Journal:** IEEE Access

**What it's about:** Reducing the computational cost of **affine motion estimation** in the VVC video coding standard. Affine ME handles non-translational motions (rotation, scaling, zooming).

**Why Affine Motion Matters:**
Standard block matching assumes objects move by pure translation (shift left/right/up/down). But in real scenes, objects also rotate, scale, and shear. Affine models capture these complex motions.

**Four-Parameter Affine Model:**
```
mv_h(x,y) = ((b_h - a_h)/w) × x + ((b_v - a_v)/w) × y + a_h
mv_v(x,y) = ((b_v - a_v)/w) × x + ((b_h - a_h)/w) × y + a_v
```
Where (a_h, a_v) and (b_h, b_v) are control point motion vectors, w is block width.

**Six-Parameter Affine Model:** Adds a third control point for independent horizontal and vertical scaling.

**Fast Algorithm (2 stages):**
1. **Early Termination:** If the parent CU's best mode is Skip (static area), skip ALL affine ME for child sub-CUs
2. **Reference Frame Reduction:** If unidirectional prediction is best, reduce reference frame search range

Uses **Bayesian posterior probability** to decide:
```
p(A|S_par) = p(S_par|A) × p(A) / p(S_par)
```
Where A = affine mode is best, S_par = parent is Skip mode.

**Results:** 37% AME time reduction with only 0.10% coding loss.

**Takeaway for Kiti:** Affine motion models are crucial for modeling complex scene motion from a moving vehicle. The Bayesian decision framework for predicting when complex motion analysis is needed vs. simple tracking could help Kiti decide when to deploy expensive analysis.

---

### Paper 2: Block-based Embedded Color Image and Video Coding
**Authors:** Nagaraj, Pearlman, Islam | **Year:** 2004 | **Venue:** SPIE

**What it's about:** Extending the SPECK wavelet-based image coder to color images (YUV 4:2:0) while maintaining embeddedness (stop decoding at any point and get a valid image).

**Key Ideas:**
- **Color-SPECK (CSPECK):** Treats all 3 color planes as one unit. Automatically allocates bits optimally among Y, U, V planes.
- **Motion-SPECK:** Intra-based video coding (no inter-frame prediction). Two modes:
  - Constant-Rate: Every frame at same bitrate
  - Constant-Distortion: Every frame at same PSNR quality

**RGB to YUV Transform:**
```
Y = 0.299R + 0.587G + 0.114B
U = 0.500R - 0.419G - 0.081B
V = -0.169R - 0.331G - 0.500B
```

**Takeaway for Kiti:** Low direct relevance to obstacle detection. However, the embedded coding approach is relevant for efficient storage/transmission of dashcam data. The YUV color space is what OpenCV uses for many processing operations.

---

### Paper 3: Run-Time Adaptive Energy-Aware Motion & Disparity Estimation ⭐ HIGH RELEVANCE
**Authors:** Bruno Zatt et al. | **Year:** 2011 | **Venue:** DAC (Design Automation Conference)

**What it's about:** Energy-efficient hardware for Motion Estimation (ME) and Disparity Estimation (DE) in multi-view video coding. ME handles temporal motion; DE handles depth from stereo cameras.

**The Energy Problem:**
- ME + DE consume up to **98% of encoding energy**
- On-chip memory for search windows consumes up to **90% of ME/DE energy**
- Rectangular search window prefetching wastes **70-95%** of the search area (fast algorithms only access a small portion)

**Key Innovations:**

1. **Dynamically Expanding Search Window:** Instead of prefetching the full rectangular search area, start small and expand based on:
   - The actual search trajectory of the algorithm
   - Motion/disparity of neighboring macroblocks
   - A **Search Map** prediction mechanism

2. **Search Map Prediction:** Predicts where the search algorithm will look based on 4 spatial predictors from neighboring blocks:
   ```
   predDiff = α × diffVariance + β × diffMotion
   ```
   Hit rate: ~80% even in worst case.

3. **Multi-Bank Power-Gating:** 16KB on-chip memory in 16 banks. Unused banks are power-gated based on predicted requirements.

**Results:**
- Off-chip memory energy: **82-96% reduction**
- On-chip memory energy: **57-75% reduction**
- ASIC: 65nm, 74mW at 1.0V, 300 MHz
- Supports: 4-view HD1080p at 30fps

**Takeaway for Kiti:** Highly relevant for two reasons:
1. **Stereo disparity estimation** is the same as depth estimation from stereo cameras - a key technique for AV obstacle distance measurement
2. The **adaptive search strategy** (expand based on motion characteristics) mirrors how AV perception should allocate more compute to high-motion regions where obstacles appear/disappear
3. Energy efficiency matters for embedded AV systems

---

### Paper 4: Efficient VVC Motion Estimation Hardware ⭐ HIGH RELEVANCE
**Authors:** Waqar Ahmad et al. | **Year:** 2024 | **Journal:** J. Real-Time Image Processing

**What it's about:** The first dedicated FPGA hardware design for VVC motion estimation, supporting the large 128×128 CTU sizes and complex QTMT partitioning.

**Key Design Choices:**

1. **Divide-and-Conquer:** Split 128×128 CTU into four 64×64 CUs → 4× smaller PE array needed
2. **64×64 Systolic PE Array:** Each PE stores a current + reference pixel, computes absolute differences
3. **Vertical Snake Scan:** PEs shift reference pixels down→up→down alternating columns. Only needs 64 new pixels per search location (high data reuse)
4. **Memory-Based SAD Adder Tree:** Hierarchical SAD: 4×4 → 8×8 → 16×16 → 32×32 → 64×64. Uses BRAM to combine 4 sub-CTU SADs.

**SAD (Sum of Absolute Differences):**
```
SAD = Σᵢ₌₀ᵂ⁻¹ Σⱼ₌₀ᴴ⁻¹ |A(i,j) - B(i,j)|
```

**Results:**
- FPGA: Xilinx Virtex 7
- 64×64 CTU: 141 fps at 1080p, **35 fps at 4K**
- 128×128 CTU: 30 fps at 1080p, 7 fps at 4K
- Supports 4309 CU partitions, 3285 unique motion vectors per CTU
- BD-Rate increase: only 0.476-1.609% (negligible quality loss)

**Takeaway for Kiti:** Demonstrates that full-search ME can be done in real-time on FPGAs at 4K resolution. The variable block-size support (4×8 to 128×128) enables adaptive-resolution motion analysis: fine blocks for obstacle boundaries, large blocks for background. FPGAs are commonly used in AV sensor processing pipelines.

---

### Paper 5: Hybrid PSO-GA Algorithm for HD Motion Estimation
**Authors:** Manal K. Jalloul, Mohamad A. Al-Alaoui | **Year:** 2014 | **Venue:** OPT-i Conference

**What it's about:** Using a hybrid Particle Swarm Optimization + Genetic Algorithm approach to solve motion estimation as a global optimization problem, specifically targeting HD video where traditional PSO fails.

**The Problem with HD Video:**
As resolution increases, the number of local minima in the search space increases dramatically because neighboring pixels become more similar. PSO particles get trapped in these local minima (premature convergence).

**PSO Basics:**
Each particle has a position (candidate motion vector) and velocity:
```
V_i(t+1) = w×V_i(t) + c₁×r₁×(P_best - X_i) + c₂×r₂×(G_best - X_i)
X_i(t+1) = X_i(t) + V_i(t+1)
```
Where:
- w = inertia weight (decreases linearly: 0.9 → 0.4)
- P_best = particle's personal best position
- G_best = swarm's global best position
- c₁, c₂ = acceleration constants
- r₁, r₂ = random numbers [0,1]

**Novel Hybrid Approach:**
1. **Smart Initialization:** Use motion vectors from 4 spatial neighbors + 5 temporal neighbors + (0,0) for static blocks. Only remaining particles are random.
2. **Adaptively-Varied Maximum Velocity:** `v_max(t) = V_max / t` - large exploration early, fine search later
3. **Fitness Function History Preservation (FFHP):** One shared array for all particles records fitness values of visited positions. Avoids redundant SAD calculations.
4. **GA Selection:** Remove 2 weakest particles (by personal best, not current fitness). Replace with offspring.
5. **Modified VPAC Crossover:** Generate children at averaged parent positions, pushed away from parents' velocity direction (increases diversity).
6. **Mutation:** Random shift within 10% of search range with probability P_m = 20%.

**Results:**
- Quality within 0.06-0.17 dB of exhaustive search at 720p
- Only 12.8 search points per macroblock at 720p (vs. 7225 for exhaustive search)
- Better than Diamond Search, 4-Step Search, and basic PSO

**Takeaway for Kiti:** The PSO-based approach to motion estimation is an alternative to gradient-based methods for finding optimal motion vectors. The **space-time correlation exploitation** in particle initialization (using neighboring and previous frame MVs) is directly applicable to initializing motion searches in Kiti's tracking pipeline.

---

### Paper 6: Efficient Optical Flow Based Motion Detection for Non-stationary Scenes ⭐ HIGHEST RELEVANCE
**Authors:** Junjie Huang et al. | **Year:** 2018 | **Venue:** arXiv (Chinese Academy of Sciences)

**What it's about:** Real-time motion detection from a moving camera using optical flow, without requiring any background model construction, training, or updating.

**Key Framework:**
1. **Mixed Optical Flow Estimation:** Use FlowNet2.0 to compute dense optical flow between current frame and frame t-k
2. **Background Optical Flow Estimation:** Model the background flow as a **quadratic function** of pixel coordinates:
   ```
   f̃ = H × p̃
   ```
   Where H is a 6×2 parameter matrix and p̃ = [x², y², xy, x, y, 1]ᵀ
3. **Constrained RANSAC (CRA):** Divide image into square pieces (S=100 pixels), randomly select P=50% of pieces, sample one point per piece. Fit quadratic model robustly.
4. **Foreground Extraction:** Compare actual flow with estimated background flow:
   ```
   M_t = {p_t | d_v > T_a}
   d_v = ||f - f̃||₁
   ```

**Adaptive Mechanisms:**
- **Adaptive Interval (AI):** Automatically adjust the frame gap k so that background flow magnitude stays around α_s = 25 pixels:
  ```
  k_{t+1} = L(α_s × k_t / mean(||f_n||₂))
  ```
  This means: slow camera → larger gap (better motion detection), fast camera → smaller gap (maintain accuracy)

- **Adaptive Threshold (AT):**
  ```
  T_a = α₁ + α₂ × mean(||f_n||₂)
  ```
  Higher threshold when camera moves fast (more background noise).

**Results:**
- **DAVIS2016 J-mean: 56.1%** (outperforms real-time methods by 24.2%)
- Speed: 51ms with fast flow estimation (20fps), 202ms with full FlowNet2.0
- No training, no model building, fully online

**Takeaway for Kiti:** This is the most directly applicable paper for Kiti's motion detection. The key insight is modeling background optical flow as a **quadratic function of coordinates** (not a simple homography). This accounts for camera rotation, translation, AND zoom simultaneously. The adaptive interval and threshold mechanisms make it robust to varying vehicle speeds. This could replace or augment Kiti's current MOG2-based approach.

---

### Paper 7: Fast Gradient Iterative Affine ME Based on Edge Detection
**Authors:** Jingping Hong et al. | **Year:** 2023 | **Journal:** Electronics (MDPI)

**What it's about:** Accelerating affine motion estimation in VVC by using Canny edge detection to speed up gradient iteration.

**Key Optimizations:**

1. **AAMVP Candidate List Optimization:** Add early-termination conditions when constructing affine motion vector candidate lists:
   - Condition_1: Is block in inter-frame mode?
   - Condition_2: Is block in affine/non-affine mode?
   - Skip remaining checks as soon as candidate list is full (length=2)

2. **Canny Edge Detection for Gradient:**
   - Standard VVC uses Sobel operator to traverse entire image for gradients
   - Replace with Canny edge detection which is more efficient for affine regions
   - Compute Canny globally on first frame, cache results
   - For subsequent frames, check if block's gradient was already computed → read from cache
   - Skip unchanged background areas, focus on motion regions

**Affine Motion Vector Iteration:**
```
mv_i(x,y) = mv_{i-1}(x,y) + A(x,y) × (d_MV^i)ᵀ
```
Where d_MV is the change in motion vector, optimized to minimize MSE:
```
MSE = (1/w×h) × Σ |P_cur(x,y) - P_ref((x,y) + mv(x,y))|²
```

**Results:**
- Overall encoding time reduced by **6.22%**
- Affine ME encoding time reduced by **24.79%**
- BD-PSNR loss: only **0.04 dB**
- BDBR increase: 0.76%

**Takeaway for Kiti:** The use of Canny edge detection to speed up gradient computation is an interesting optimization. In Kiti's context, focusing motion analysis on edge regions (where objects are) rather than uniform regions (sky, road surface) could significantly reduce computation.

---

## Key Concepts & Techniques

### 1. Block Matching Motion Estimation
The foundation of motion estimation in video coding. For each block in the current frame, find the best matching block in the reference frame.

**Search Strategies (from simple to complex):**
| Method | Search Points | Quality |
|--------|--------------|---------|
| Full/Exhaustive Search | All (e.g., 7225) | Optimal |
| Diamond Search | ~20-30 | Good for smooth motion |
| 4-Step Search | ~20-25 | Good for moderate motion |
| PSO-based (Paper 5) | ~13 | Near-optimal |
| Proposed Hybrid PSO | ~13 | Closest to optimal |

### 2. Translational vs. Affine Motion Models
- **Translational:** Block moves by (dx, dy). Simple, fast. Handles most cases.
- **4-parameter Affine:** Handles rotation + uniform scaling + translation. Uses 2 control points.
- **6-parameter Affine:** Handles rotation + non-uniform scaling + shear + translation. Uses 3 control points.

### 3. Quadratic Background Flow Model
Instead of assuming background flow is a simple homography (8 parameters), model it as a quadratic function (12 parameters):
```
u(x,y) = a₁x² + a₂y² + a₃xy + a₄x + a₅y + a₆
v(x,y) = b₁x² + b₂y² + b₃xy + b₄x + b₅y + b₆
```
This captures camera translation, rotation, AND zoom effects in a single model.

### 4. Disparity Estimation (Stereo Depth)
Same as motion estimation but between left and right camera views instead of consecutive frames. The "motion vector" becomes a "disparity vector" which encodes depth:
```
depth = (focal_length × baseline) / disparity
```
Where baseline is the distance between stereo cameras.

### 5. Rate-Distortion Optimization
The fundamental trade-off in video coding:
```
J = D + λ × R
```
Where D = distortion (quality loss), R = bit rate (data size), λ = trade-off parameter.
In AV context: trade-off between detection accuracy and processing speed.

---

## Mathematical Foundations

### SAD - Sum of Absolute Differences
```
SAD = Σᵢ Σⱼ |Current(i,j) - Reference(i+dx, j+dy)|
```
**Fast to compute, widely used.** Lower SAD = better match.

### MSE - Mean Squared Error
```
MSE = (1/N) × Σ(Current - Reference)²
```
**Penalizes large errors more.** Used in affine ME.

### PSNR - Peak Signal-to-Noise Ratio
```
PSNR = 10 × log₁₀(255² / MSE) dB
```
Higher = better quality. Typical range: 25-45 dB.

### PSO Velocity Update
```
V_i(t+1) = w×V_i(t) + c₁×r₁×(P_best - X_i) + c₂×r₂×(G_best - X_i)
```
Three components: inertia (keep going), cognitive (personal best), social (swarm best).

### Affine Transformation (4-parameter)
```
| x' |   | a  b | | x |   | tx |
|    | = |      | |   | + |    |
| y' |   | -b a | | y |   | ty |
```
Where a = s×cos(θ), b = s×sin(θ), s = scale, θ = rotation angle.

### Quadratic Optical Flow Model
```
u(x,y) = H₁ × [x², y², xy, x, y, 1]ᵀ  (horizontal flow)
v(x,y) = H₂ × [x², y², xy, x, y, 1]ᵀ  (vertical flow)
```

---

## Relevance to Kiti Project

### Direct Applicability Map

| Paper | Kiti Component | How It Applies |
|-------|---------------|----------------|
| Paper 1 (Affine ME) | Motion detection | Model complex obstacle motion (rotation, scaling) |
| Paper 3 (Energy-Aware ME/DE) | Distance estimation | Stereo disparity = depth estimation |
| Paper 4 (VVC ME Hardware) | Performance | FPGA acceleration for real-time 4K processing |
| Paper 5 (Hybrid PSO) | Object tracking | Smart initialization of motion search using neighbors |
| Paper 6 (Optical Flow Detection) | Core pipeline | Quadratic flow model for moving camera motion detection |
| Paper 7 (Canny + Affine ME) | Optimization | Focus computation on edge regions, skip uniform areas |

### Recommended Integration Points

1. **Replace/Augment MOG2 with Quadratic Flow Model (Paper 6)**
   - Current: MOG2 background subtraction (assumes static camera)
   - Proposed: Estimate background flow as quadratic function, subtract from total flow
   - Benefit: Works with moving camera, no model training needed

2. **Add Affine Motion Analysis for Complex Objects (Paper 1)**
   - Current: Only translational motion vectors
   - Proposed: Use affine models for objects undergoing rotation/scaling
   - Benefit: Better tracking of turning vehicles, approaching/receding objects

3. **Smart Computation Allocation (Papers 3, 7)**
   - Current: Uniform processing across entire frame
   - Proposed: Allocate more compute to edge regions and high-motion areas
   - Benefit: Faster processing, same detection quality

4. **Stereo Depth Estimation (Paper 3)**
   - Current: Monocular focal-length-based distance estimation
   - Proposed: Add stereo camera support using disparity estimation
   - Benefit: More accurate depth, especially at closer ranges

---

## Further Reading & References

### Foundational Algorithms
- **Farneback Dense Optical Flow:** Farneback, G. (2003). "Two-frame motion estimation based on polynomial expansion." Already used in Kiti.
- **FlowNet2.0:** Ilg, E. et al. (2017). "FlowNet 2.0: Evolution of optical flow estimation with deep networks." https://github.com/NVIDIA/flownet2-pytorch
- **RAFT Optical Flow:** Teed, Z. & Deng, J. (2020). "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow." State-of-the-art. https://github.com/princeton-vl/RAFT

### Video Coding Standards
- **VVC/H.266:** Latest video coding standard. Contains state-of-the-art ME tools.
- **VTM Reference Software:** https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM

### Optimization Techniques
- **Particle Swarm Optimization:** Kennedy, J. & Eberhart, R. (1995). "Particle swarm optimization."
- **RANSAC:** Fischler, M.A. & Bolles, R.C. (1981). "Random sample consensus."

### Hardware Acceleration
- **OpenCV CUDA module:** GPU-accelerated computer vision functions
- **TensorRT:** NVIDIA's deep learning inference optimizer
- **Vitis AI:** Xilinx's AI inference on FPGAs

### Datasets
- **KITTI:** http://www.cvlibs.net/datasets/kitti/ - Autonomous driving benchmark with stereo, optical flow, and tracking ground truth
- **DAVIS2016:** https://davischallenge.org/ - Video object segmentation with moving camera
- **Xiph Test Sequences:** https://media.xiph.org/video/derf/ - Standard video coding test sequences
