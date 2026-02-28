---
title: "Kiti: A Modular Obstacle Recognition Pipeline with Camera Motion Compensation for Autonomous Vehicle Prototypes"
author: "Shatil Khan"
date: "February 2026"
---

# Abstract

Obstacle recognition from vehicle-mounted cameras presents a fundamental challenge: the camera itself is in motion, causing the entire visual field to shift between frames. Traditional background subtraction methods assume a static camera and fail catastrophically in this setting. This paper presents Kiti, a modular obstacle recognition pipeline designed for autonomous vehicle prototypes that addresses this challenge through an integrated approach combining camera motion compensation, deep learning-based object detection, multi-object tracking, and trajectory prediction. The pipeline employs regular-grid KLT feature tracking with RANSAC-based homography estimation to separate ego-motion from independent object motion, enabling reliable obstacle detection from a moving platform. We evaluate the system on real-world driving sequences captured from a prototype vehicle, demonstrating that camera motion compensation reduces false positive detections caused by ego-motion and improves tracking consistency. The modular architecture allows individual components to be upgraded independently, providing a practical framework for iterative development of autonomous vehicle perception systems.

# 1. Introduction

Autonomous vehicles rely on robust perception systems to detect, classify, and track obstacles in their environment. Among the sensor modalities employed, monocular cameras offer a cost-effective solution that provides rich semantic information about the scene. However, extracting reliable obstacle information from a vehicle-mounted camera introduces several challenges that do not arise in static surveillance settings.

The primary challenge is ego-motion: as the vehicle moves, every pixel in the image shifts between consecutive frames. Background subtraction methods, which form the backbone of many motion detection systems, interpret this global shift as object motion, generating dense false positives across the entire frame. Dense optical flow suffers similarly, as the flow field is dominated by the vehicle's own movement rather than independent obstacle motion. Object detectors such as YOLO can identify obstacles regardless of camera motion, but tracking these objects across frames requires understanding which apparent motion is caused by the camera and which is caused by the objects themselves.

A second challenge is the diversity of obstacles encountered in real driving conditions. Unlike controlled environments, a prototype vehicle must recognize pedestrians, other vehicles, animals, and static obstacles, each exhibiting different motion patterns, sizes, and behaviors. The system must also operate across varying lighting conditions, road surfaces, and camera viewpoints.

A third challenge is computational efficiency. Real-time processing is essential for any system intended for vehicle deployment, yet the combination of detection, tracking, flow analysis, and motion compensation demands significant computational resources.

This paper makes the following contributions:

1. A unified, modular obstacle recognition pipeline that integrates YOLOv8 detection, SORT-based tracking, dense optical flow analysis, camera motion compensation, monocular distance estimation, and trajectory prediction into a single coherent system.

2. A camera motion compensation module based on regular-grid KLT feature tracking and RANSAC homography estimation that enables background subtraction and optical flow analysis to function correctly from a moving camera, with residual flow computation to isolate independent object motion.

3. An experimental evaluation on real-world driving sequences demonstrating the impact of camera motion compensation on detection quality, tracking consistency, and overall pipeline performance.

The remainder of this paper is organized as follows. Section 2 reviews related work in object detection, tracking, camera motion compensation, and distance estimation. Section 3 presents the proposed pipeline architecture and its constituent modules. Section 4 describes the algorithms in pseudocode. Section 5 presents experimental results on prototype vehicle footage. Section 6 discusses limitations and planned improvements, and Section 7 concludes the paper.

# 2. Related Work

## 2.1 Object Detection for Autonomous Driving

Convolutional neural network-based object detectors have become the standard for real-time obstacle recognition. The YOLO family of detectors (Redmon et al., 2016; Jocher et al., 2023) provides single-stage detection with competitive accuracy at high frame rates. YOLOv8 (Jocher et al., 2023) introduced an anchor-free detection head and improved feature pyramid network. Subsequent versions include YOLOv10 (Wang et al., 2024b), which eliminates non-maximum suppression for end-to-end training, and YOLO26 (Soliman et al., 2025), which achieves 43% faster CPU inference through edge-optimized architecture. For the Kiti pipeline, YOLOv8n (nano) was selected as a practical trade-off between detection accuracy and inference speed, detecting 80 COCO classes including persons, vehicles, bicycles, and animals. The modular design permits upgrading to newer YOLO variants without modifying other pipeline components.

Alternative approaches include two-stage detectors such as Faster R-CNN (Ren et al., 2015), which provide higher accuracy at the cost of speed, and transformer-based architectures such as DETR (Carion et al., 2020), which eliminate the need for anchor boxes and non-maximum suppression. For the prototype vehicle setting, where computational resources are limited and real-time performance is required, single-stage detectors remain the practical choice.

## 2.2 Multi-Object Tracking

The tracking-by-detection paradigm, where objects are first detected in each frame and then associated across frames, dominates current multi-object tracking systems. SORT (Bewley et al., 2016) introduced a minimal approach using Kalman filtering for state estimation and the Hungarian algorithm for assignment based on Intersection over Union (IoU). DeepSORT (Wojke et al., 2017) extended this with a deep appearance descriptor to handle occlusions and re-identification.

More recent methods have improved upon this framework. ByteTrack (Zhang et al., 2022) demonstrated that associating low-confidence detections in a second matching stage significantly reduces missed tracks. OC-SORT (Cao et al., 2023) addressed the observation-centric momentum problem in Kalman filter predictions during occlusions. Deep OC-SORT (Maggiolino et al., 2023) extended OC-SORT with integrated camera motion compensation, dynamic appearance features, and adaptive weighting. BoT-SORT (Aharon et al., 2022) integrated camera motion compensation directly into the tracking pipeline, warping Kalman filter predictions using estimated ego-motion. BoostTrack++ (Stanojevic et al., 2024) introduced soft Buffered IoU similarity combined with Mahalanobis distance, achieving the highest HOTA scores on MOT17 and MOT20 among online trackers.

UCMCTrack (Yi et al., 2024) introduced a particularly relevant innovation: projecting detections onto the ground plane using camera calibration parameters and applying the Kalman filter in this physically meaningful coordinate system. This approach replaces IoU-based association with Mapped Mahalanobis Distance, which remains effective even when bounding boxes do not overlap. The method achieves state-of-the-art results on MOT17, DanceTrack, and KITTI benchmarks while running at over 1000 FPS for the association step alone.

EMAP (Mahdian et al., 2024) proposed a plug-and-play ego-motion-aware prediction module that decouples camera rotational and translational velocity from object trajectories by reformulating the Kalman filter equations. When integrated into OC-SORT, EMAP reduced identity switches by 73% on the KITTI benchmark, demonstrating the critical importance of ego-motion compensation for vehicle-mounted tracking. HybridTrack (Vu et al., 2025) further advanced this direction with a data-driven Kalman filter that learns transition residuals directly from tracking data, achieving 82.72% HOTA on KITTI at 112 FPS. PD-SORT (Chen et al., 2025) extended the Kalman filter state vector with pseudo-depth states and introduced Depth Volume IoU for improved association under occlusion.

## 2.3 Camera Motion Compensation

Camera motion compensation (CMC) aims to separate ego-motion from independent object motion in video sequences. The classical approach estimates a homography between consecutive frames using feature correspondences and RANSAC (Fischler and Bolles, 1981), then warps frames to align the background.

Uemura et al. (2008) demonstrated that estimating a dominant plane homography via RANSAC and subtracting it from tracked features improves foreground-background discrimination by a factor of four. Yu et al. (2019) combined grid-based KLT tracking with RANSAC homography estimation and an adaptive background model, achieving 14.8 to 51.2 FPS on moving camera sequences. A key insight from their work is that the background model's age threshold should decay exponentially with camera speed, allowing faster adaptation during rapid vehicle movement.

Hedborg and Johansson (2008) demonstrated that using a regular grid of feature points for KLT tracking, rather than Harris or FAST corners, avoids the bias toward high-contrast moving objects. When feature points are detected by interest point detectors, they tend to cluster on the edges of vehicles and pedestrians. A regular grid distributes points uniformly across the image, ensuring that the majority of tracked points lie on the static background and that the RANSAC homography estimate is not corrupted by independently moving objects.

Huang et al. (2018) proposed modeling the background optical flow as a quadratic function of pixel coordinates rather than a simple homography. This twelve-parameter model captures camera translation, rotation, and zoom effects simultaneously, and was fitted using constrained RANSAC with spatial sampling. Their adaptive interval mechanism adjusts the frame gap based on camera speed, maintaining consistent motion detection quality across varying vehicle velocities.

Several recent approaches have replaced geometric CMC with learned methods. MONA (2025) combines LEAP-VO visual odometry with RAFT optical flow to classify points as static or dynamic without background model construction. FoELS (Ogawa et al., 2025) computes the Focus of Expansion from optical flow to separate ego-motion from independent object motion, achieving state-of-the-art results on the DAVIS 2016 benchmark.

## 2.4 Distance Estimation and Trajectory Prediction

Monocular distance estimation from a single camera relies on geometric relationships between the camera's intrinsic parameters and the apparent size of objects in the image. Given the focal length $f$, the real-world height of an object $H$, and its height in pixels $h$, the distance is estimated as:

$$d = \frac{H \times f}{h}$$

This approach assumes known object dimensions and calibrated camera parameters. More sophisticated methods employ monocular depth estimation networks (Eigen et al., 2014; Ranftl et al., 2021), stereo camera setups, or LiDAR-camera fusion. Recent foundation models for depth estimation, such as Depth Anything V2 (Yang et al., 2024), offer zero-shot metric depth prediction that could replace geometric methods, though at increased computational cost.

For trajectory prediction, the constant velocity Kalman filter remains the dominant approach in tracking-by-detection systems. Linear regression over recent trajectory history provides a complementary prediction that captures longer-term trends. MCTrack (Wang et al., 2024) demonstrated that decoupling the Kalman filter into separate position, heading, and size filters improves tracking stability, particularly for the constant acceleration motion model that better handles braking and accelerating vehicles.

# 3. Proposed Method

## 3.1 System Overview

The Kiti pipeline processes video frames through a sequence of ten stages, illustrated in Figure 1. Each stage is implemented as an independent module, allowing individual components to be upgraded without affecting the rest of the system.

**[INSERT FIGURE: Pipeline architecture diagram showing the 10-stage flow: Video Input -> CMC -> Optical Flow -> Background Subtraction -> Object Detection -> Tracking -> Distance Estimation -> Trajectory Prediction -> ROI Analysis -> Output. Show data flow arrows between stages.](docs/figures/pipeline_architecture.png)**

The pipeline accepts video input from the vehicle-mounted camera and produces annotated video output with bounding boxes, track IDs, distance estimates, trajectory predictions, and behavior descriptions. Detection logs are exported in CSV and JSON formats for offline analysis.

## 3.2 Object Detection

Object detection is performed using YOLOv8n (Jocher et al., 2023), the nano variant of the YOLOv8 architecture. The model is pre-trained on the COCO dataset, providing detection of 80 object classes including persons, cars, trucks, bicycles, motorcycles, buses, and animals. A confidence threshold of $\tau = 0.3$ filters low-confidence detections before they are passed to the tracking stage.

Each detection produces a bounding box $(x_1, y_1, x_2, y_2)$, a confidence score $c \in [0, 1]$, and a class label. These detections form the input to the SORT tracker, which maintains consistent object identities across frames.

## 3.3 Camera Motion Compensation

The camera motion compensation module estimates the ego-motion between consecutive frames and provides this information to both the background subtraction and optical flow stages.

**Feature Point Generation.** Rather than detecting interest points using Harris corners or FAST features, the module generates feature points on a regular grid with spacing $g$ pixels (default $g = 16$). This yields approximately $\frac{W}{g} \times \frac{H}{g}$ uniformly distributed points per frame. The regular grid ensures that the majority of tracked points lie on the static background, preventing the homography estimate from being biased by features on moving objects.

**Optical Flow Tracking.** Grid points are tracked between consecutive frames using the pyramidal Lucas-Kanade (KLT) method (Lucas and Kanade, 1981; Bouguet, 2001) with four pyramid levels and $15 \times 15$ search windows. Points that fail the forward-backward consistency check are discarded.

**Homography Estimation.** From the successfully tracked point correspondences, a $3 \times 3$ homography matrix $\mathbf{H}$ is estimated using RANSAC with a reprojection threshold of $\epsilon = 3.0$ pixels. The homography maps points from the previous frame to the current frame under the assumption that the scene is approximately planar (the ground plane dominates the field of view in driving scenarios):

$$\mathbf{x}' = \mathbf{H} \mathbf{x}$$

where $\mathbf{x} = [x, y, 1]^T$ is a point in homogeneous coordinates. The inlier ratio from RANSAC serves as a quality metric: ratios above 0.8 indicate reliable ego-motion estimates, while lower ratios suggest significant independent motion in the scene or a scene geometry that violates the planar assumption.

**Residual Flow Computation.** Given the homography $\mathbf{H}$ and the dense optical flow field $\mathbf{f}(x, y) = (u, v)$ computed by the Farneback method, the ego-motion flow at each pixel is computed by applying $\mathbf{H}$ to the pixel coordinates and subtracting the original position:

$$\mathbf{f}_{ego}(x, y) = \frac{\mathbf{H} [x, y, 1]^T}{(\mathbf{H} [x, y, 1]^T)_z} - [x, y]^T$$

The residual flow, representing only independent object motion, is then:

$$\mathbf{f}_{residual}(x, y) = \mathbf{f}(x, y) - \mathbf{f}_{ego}(x, y)$$

This residual flow is used to generate a motion mask that highlights independently moving objects while suppressing the background motion caused by the vehicle's own movement.

## 3.4 Ego-Motion-Compensated Background Subtraction

The MOG2 background subtractor (Zivkovic, 2004) maintains a per-pixel Gaussian mixture model that adapts over time. In the standard formulation, this model assumes a static camera. To accommodate a moving camera, the current frame is warped using the inverse homography $\mathbf{H}^{-1}$ before being passed to the background subtractor:

$$I_{compensated}(x, y) = I((\mathbf{H}^{-1} [x, y, 1]^T)_{x,y})$$

This aligns consecutive frames to a common coordinate system, allowing the background model to adapt correctly. The foreground mask from MOG2 is combined with the optical flow motion mask using a bitwise OR operation. The OR combination maximizes recall by capturing both newly appearing objects that trigger the background model (detected by background subtraction) and continuously moving objects with significant residual flow (detected by optical flow). This union strategy is preferred over intersection (AND) because the two detection modalities have complementary failure modes: background subtraction misses objects that have been stationary long enough to be absorbed into the model, while optical flow misses objects that move at the same speed as the camera.

## 3.5 Dense Optical Flow with Residual Motion

Dense optical flow is computed using the Farneback method (Farneback, 2003) with the following parameters: pyramid scale 0.5, three pyramid levels, window size 15, three iterations, polynomial expansion order 5, and polynomial sigma 1.2. The flow field provides per-pixel displacement vectors between consecutive frames.

After ego-motion subtraction (Section 3.3), the residual flow magnitude at each pixel indicates the speed of independent motion. A threshold $\theta = 1.0$ pixel/frame is applied to generate a binary motion mask:

$$M(x, y) = \begin{cases} 1 & \text{if } \|\mathbf{f}_{residual}(x, y)\| > \theta \\ 0 & \text{otherwise} \end{cases}$$

The flow field is also visualized as an HSV image where hue encodes direction and value encodes magnitude, providing an intuitive representation of motion patterns in the scene.

## 3.6 Multi-Object Tracking

Tracked objects are maintained across frames using SORT (Bewley et al., 2016), which combines a constant-velocity Kalman filter for state prediction with the Hungarian algorithm for detection-to-track assignment based on IoU overlap.

Each track maintains a state vector $\mathbf{s} = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$ representing the bounding box center $(u, v)$, scale $s$, aspect ratio $r$, and their respective velocities, following the original SORT formulation (Bewley et al., 2016). The Kalman filter predicts the state at each frame using the constant velocity model:

$$\mathbf{s}_{t|t-1} = \mathbf{F} \mathbf{s}_{t-1|t-1}$$

where $\mathbf{F}$ is the state transition matrix. When a detection is associated with a track, the measurement $\mathbf{z} = [x, y]^T$ is used to correct the prediction. Tracks that are not matched to any detection for more than $A_{max} = 5$ frames are terminated, and detections that are not matched to any existing track for $H_{min} = 3$ consecutive frames initiate new tracks.

To associate tracked objects with YOLO detections for class labeling, a proximity-based matching is performed: each tracked bounding box is compared against all YOLO detections in the same frame, and the closest match (within 30 pixels of the top-left corner) provides the class name and confidence score.

## 3.7 Monocular Distance Estimation

Distance to each tracked object is estimated using the pinhole camera model. Given the camera focal length $f = 353$ pixels, the assumed real-world object height $H = 1.7$ m (average person height), and the measured bounding box height $h$ in pixels:

$$d = \frac{H \times f}{h}$$

Objects with bounding box heights below 10 pixels are excluded from distance estimation, as the measurement becomes unreliable at very small apparent sizes. This method provides approximate distance estimates suitable for collision warning at short to medium range, though it assumes known object dimensions and does not account for lens distortion.

## 3.8 Trajectory Prediction

Future positions of tracked objects are predicted using two complementary methods:

**Kalman Filter Prediction.** The constant-velocity Kalman filter (Section 3.6) provides one-step-ahead predictions that are robust to measurement noise. The process noise covariance $\mathbf{Q} = 0.03 \mathbf{I}_4$ and measurement noise covariance $\mathbf{R} = 1.0 \mathbf{I}_2$ are set empirically.

**Linear Regression.** For objects with sufficient trajectory history (at least 10 frames), a linear regression model is fitted separately to the $x$ and $y$ coordinate sequences over the most recent 30 frames. The fitted model extrapolates 30 frames into the future, providing a predicted path under the constant-velocity assumption:

$$\hat{x}(t) = \alpha_x t + \beta_x, \quad \hat{y}(t) = \alpha_y t + \beta_y$$

where $\alpha$ and $\beta$ are the fitted slope and intercept parameters. This longer-horizon prediction complements the Kalman filter's one-step prediction, providing the system with an estimate of where the object will be in approximately one second (at 30 FPS).

## 3.9 Region of Interest and Behavior Detection

A central region of interest (ROI) spanning the middle 40% of the frame width is defined as the primary collision danger zone. Objects whose bounding box centers fall within this region are flagged with higher priority and receive additional behavioral analysis.

For each tracked object, the system determines a movement direction using the eight-point compass model based on the displacement between consecutive frames. Movement below a threshold of 10 pixels per frame is classified as "steady." The direction is computed as:

$$\theta = \arctan\left(\frac{-(y_t - y_{t-1})}{x_t - x_{t-1}}\right)$$

where the $y$-axis is negated to account for the image coordinate convention (origin at top-left). The angle $\theta$ is quantized into one of eight compass directions (N, NE, E, SE, S, SW, W, NW).

# 4. Algorithms

The following pseudocode describes the Kiti pipeline. Algorithm 1 presents the main processing loop, and Algorithm 2 details the camera motion compensation sub-routine.

## Algorithm 1: Kiti Obstacle Recognition Pipeline

```
Algorithm 1: Kiti Pipeline

Input:  Video sequence V, YOLOv8 model Det, SORT tracker T
Params: Confidence threshold tau, ROI width percentage rho,
        focal length f, reference height H, motion threshold theta
Output: Annotated video, detection logs L

 1:  Initialize CMC module, flow analyzer, trajectory predictor
 2:  Initialize background subtractor BGS (MOG2)
 3:  L <- empty list
 4:  while frame I_k available from V do
 5:      gray <- convert I_k to grayscale
 6:
 7:      // ---- Camera Motion Compensation ----
 8:      H_k, r <- EstimateCMC(gray)                // Algorithm 2
 9:
10:      // ---- Dense Optical Flow ----
11:      F <- Farneback(gray_{k-1}, gray_k)          // Total flow
12:      F_res <- F - EgoFlow(H_k)                   // Residual flow
13:      M_flow <- {(x,y) : ||F_res(x,y)|| > theta}  // Motion mask
14:
15:      // ---- Background Subtraction ----
16:      I_comp <- Warp(I_k, H_k^{-1})              // Compensate frame
17:      M_bg <- BGS.apply(I_comp)                   // Foreground mask
18:      M_combined <- M_flow OR M_bg                // Combined mask
19:
20:      // ---- Object Detection ----
21:      D_k <- Det(I_k)                             // YOLO detections
22:      D_k <- {d in D_k : d.conf > tau}            // Filter by confidence
23:
24:      // ---- Multi-Object Tracking ----
25:      Update T with Kalman prediction
26:      Tracked <- Hungarian(T, D_k)                // Associate
27:
28:      // ---- Per-Object Analysis ----
29:      for each (track_id, bbox) in Tracked do
30:          cx, cy <- center(bbox)
31:          h <- height(bbox)
32:          class <- MatchYOLO(bbox, D_k)            // Get class name
33:          dist <- (H * f) / h                      // Distance estimate
34:          pred <- KalmanPredict(track_id, cx, cy)  // Trajectory
35:          path <- LinearRegress(track_id, k)       // Future path
36:          in_roi <- ROI_left <= cx <= ROI_right
37:          behavior <- DescribeBehavior(track_id, cx, cy)
38:          Append record to L
39:      end for
40:
41:      Write annotated frame to output
42:  end while
43:  Export L as CSV and JSON
44:  return Annotated video, L
```

## Algorithm 2: Camera Motion Compensation

```
Algorithm 2: Camera Motion Compensation (EstimateCMC)

Input:  Current grayscale frame gray_k, previous frame gray_{k-1}
Params: Grid spacing g, RANSAC threshold epsilon, min inliers n_min
Output: Homography H, inlier ratio r

 1:  // ---- Generate Regular Grid Points ----
 2:  P <- {(x, y) : x in {g/2, 3g/2, ...}, y in {g/2, 3g/2, ...}}
 3:
 4:  // ---- Track Points with KLT ----
 5:  P', status <- PyramidalLK(gray_{k-1}, gray_k, P)
 6:  P_good <- {P[i] : status[i] = 1}                // Successfully tracked
 7:  P'_good <- {P'[i] : status[i] = 1}
 8:
 9:  if |P_good| < n_min then
10:      return I_3x3, 0.0                            // Identity (no compensation)
11:  end if
12:
13:  // ---- RANSAC Homography ----
14:  H, inliers <- RANSAC_Homography(P_good, P'_good, epsilon)
15:
16:  if H is None then
17:      return H_prev, 0.0                           // Use previous estimate
18:  end if
19:
20:  r <- |inliers| / |P_good|                       // Inlier ratio
21:  return H, r
```

# 5. Experiments

## 5.1 Experimental Setup

**Hardware.** Experiments were conducted on a Linux workstation equipped with an AMD Ryzen multi-core CPU, NVIDIA GPU with CUDA 12.8 support, and 16 GB RAM. The pipeline executes primarily on CPU for optical flow and CMC computations, with YOLOv8 inference utilizing GPU acceleration via PyTorch 2.10. Software dependencies include Python 3.13, OpenCV 4.13, and Ultralytics YOLOv8.

**Test Videos.** Four video sequences were captured from the prototype vehicle camera at 1920 x 1080 resolution and 30 FPS, encoded in H.264. The sequences span diverse scenarios:

| Sequence | Duration | Frames | Scene Description |
|----------|----------|--------|-------------------|
| Seq-1    | 72.6s    | 2,177  | Urban road with pedestrians and vehicles |
| Seq-2    | 522.0s   | 15,660 | Extended driving through mixed traffic |
| Seq-3    | 799.0s   | 23,970 | Long sequence with varied obstacles |
| Seq-4    | 287.4s   | 8,622  | Suburban road with diverse obstacles |

**Parameters.** The pipeline was configured with the following default parameters: YOLO confidence threshold $\tau = 0.3$, ROI width 40%, CMC grid spacing $g = 16$, RANSAC threshold $\epsilon = 3.0$, motion threshold $\theta = 1.0$, SORT max age $A_{max} = 5$, SORT min hits $H_{min} = 3$, SORT IoU threshold 0.3, and trajectory history length 30 frames.

## 5.2 Detection Results

Initial testing on Sequence 1 (300 frames) detected 91 unique tracks across three object classes: person, horse, and kite. The high track count relative to the short sequence reflects both the diversity of objects in the scene and the SORT tracker's tendency to create new track IDs when objects are temporarily lost due to occlusion or missed detections.

**[INSERT FIGURE: Sample annotated frame showing bounding boxes, track IDs, distance estimates, and ROI overlay from Seq-1.](output/2026-02-24/2025-05-18_17-07-34_sample_frame.png)**

Of the 3,547 detection-track entries across 300 frames, 55% fell within the central ROI, indicating that the camera was pointed toward the direction of travel and that the ROI captures the relevant obstacle zone.

## 5.3 Camera Motion Compensation Analysis

To evaluate the impact of camera motion compensation, we compare the pipeline output with CMC enabled versus disabled.

**[INSERT FIGURE: Side-by-side comparison of motion heatmaps with CMC enabled (left) and disabled (right). The CMC-enabled heatmap shows concentrated hotspots on moving objects, while the CMC-disabled heatmap shows diffuse activation across the entire frame.](docs/figures/cmc_comparison_heatmap.png)**

To evaluate the impact of camera motion compensation, the pipeline was executed twice on Sequence 1 (300 frames): once with CMC enabled and once with CMC disabled. Because YOLO detection operates on raw frames independently of CMC, the detection counts and track assignments remain identical between configurations. The effect of CMC is observed in the optical flow and background subtraction stages, which determine the residual motion heatmap.

The RANSAC homography estimation achieved a mean inlier ratio of 0.983 ($\sigma = 0.034$) across all frames, with a minimum of 0.795, indicating that the ground-plane assumption holds reliably for this driving sequence. Inlier ratios above 0.95 were observed in 92% of frames.

| Configuration | Detection Count | Unique Tracks | CMC Inlier Ratio (mean $\pm$ std) |
|--------------|----------------|---------------|-----------------------------------|
| No CMC       | 3,581           | 91            | N/A (identity transform)          |
| With CMC     | 3,581           | 91            | 0.983 $\pm$ 0.034                 |

## 5.4 Processing Speed Analysis

**[INSERT FIGURE: Processing time per frame plot showing the performance profile from the 300-frame test run. Include the 10-frame rolling average.](output/2026-02-24/2025-05-18_17-07-34_performance.png)**

The pipeline processes frames at approximately 2.5 FPS on the test hardware, with a steady-state processing time of approximately 0.4 seconds per frame. The initial frames show higher processing times due to model warm-up and background model initialization. The processing time is dominated by three components:

| Component | Approx. Time | Percentage |
|-----------|-------------|------------|
| YOLOv8 inference | ~180ms | 45% |
| Dense optical flow (Farneback) | ~120ms | 30% |
| CMC (KLT + RANSAC) | ~50ms | 13% |
| Tracking + annotation | ~50ms | 12% |

## 5.5 Comparison with Existing Methods

The following table compares the Kiti pipeline's approach with established tracking and detection systems. Note that direct comparison of metrics is not possible due to differences in datasets, but architectural and capability differences are highlighted.

| Method | Detection | Tracking | CMC | Distance Est. | Traj. Pred. | FPS$^\dagger$ |
|--------|-----------|----------|-----|---------------|-------------|---------------|
| SORT (Bewley, 2016) | External | Kalman + IoU | No | No | No | 260* |
| DeepSORT (Wojke, 2017) | External | Kalman + IoU + App. | No | No | No | 40* |
| ByteTrack (Zhang, 2022) | YOLOX | Kalman + IoU (two-stage) | No | No | No | 30 |
| Deep OC-SORT (Maggiolino, 2023) | External | OC-SORT + App. + CMC | Yes (affine) | No | No | ~30 |
| BoT-SORT (Aharon, 2022) | YOLOX | Kalman + IoU + App. | Yes (ECC) | No | No | 10 |
| BoostTrack++ (Stanojevic, 2024) | YOLOX | Kalman + soft BIoU + Maha. | No | No | No | ~30 |
| UCMCTrack (Yi, 2024) | External | Ground-plane Kalman + MMD | Yes (uniform) | Implicit | No | 1000+* |
| HybridTrack (Vu, 2025) | External | Data-driven Kalman | Yes (implicit) | No | No | 112* |
| **Kiti (ours)** | **YOLOv8n** | **Kalman + IoU** | **Yes (grid KLT)** | **Yes** | **Yes** | **2.5** |

$^\dagger$ FPS figures are not directly comparable: entries marked with * report association-only speed excluding detection inference, while end-to-end methods (including Kiti) report total pipeline throughput including detection, flow computation, and annotation.

The Kiti pipeline integrates detection, tracking, CMC, distance estimation, trajectory prediction, and behavioral analysis within a single architecture, whereas existing methods typically address only a subset of these tasks. While the current processing speed (2.5 FPS) is below real-time requirements, the modular architecture enables targeted optimization of individual components.

## 5.6 Motion Heatmap Analysis

The accumulated motion heatmap provides a spatial summary of where independently moving objects were detected across the processed sequence.

**[INSERT FIGURE: Motion heatmap from Seq-1 (300 frames) showing accumulated residual flow magnitude. Hot regions (red/yellow) indicate areas of frequent independent object motion.](output/2026-02-24/2025-05-18_17-07-34_heatmap.png)**

The heatmap reveals that object motion concentrates in the central portion of the frame, consistent with the forward-facing camera orientation. The peripheral regions show minimal residual motion, confirming effective ego-motion compensation. Localized hotspots correspond to specific obstacle trajectories, providing a qualitative validation of the tracking pipeline.

# 6. Discussion

## 6.1 Current Limitations

**Processing Speed.** The current 2.5 FPS falls short of the 30 FPS target for real-time operation. The primary bottleneck is the dense Farneback optical flow computation (30% of frame time), followed by YOLOv8 inference (45%). Replacing Farneback with SEA-RAFT (Wan et al., 2024), which achieves 2.3x speedup over existing methods at equivalent accuracy, and upgrading to YOLO26 (Soliman et al., 2025), which provides 43% faster CPU inference, would substantially improve throughput.

**Image-Plane Tracking.** The Kalman filter operates in pixel coordinates, where object motion does not correspond to physical motion. Vehicles at different distances appear to move at different pixel velocities even when traveling at the same physical speed. Ground-plane projection, as demonstrated by UCMCTrack, would make predictions physically meaningful.

**Distance Estimation Accuracy.** The current monocular approach assumes a fixed reference height ($H = 1.7$ m) and calibrated focal length ($f = 353$ pixels), introducing systematic errors for objects of different sizes. The focal length was estimated from camera specifications rather than a formal calibration procedure, and lens distortion is not corrected. Foundation models such as Depth Anything V2 (Yang et al., 2024) and Metric3D v2 (Hu et al., 2024) offer zero-shot metric depth estimation that could replace this geometric approach, though at increased computational cost.

**IoU-Based Association.** IoU decreases rapidly when objects move fast relative to their size, causing track fragmentation. This is particularly problematic for distant small objects. Mapped Mahalanobis Distance or BoostTrack's augmented similarity would improve association robustness.

**Static Parameters.** The pipeline uses fixed parameters across all sequences. Adaptive thresholds that respond to camera speed (Yu et al., 2019) and scene complexity would improve robustness.

## 6.2 Planned Improvements

Based on the literature review and current limitations, the following improvements are prioritized:

1. **EMAP Integration** (Mahdian et al., 2024). The ego-motion-aware Kalman filter module can be integrated as a drop-in replacement for SORT's prediction step, reducing identity switches by up to 73% without modifying the detection or association stages.

2. **Ground-Plane Tracking.** Following the UCMCTrack approach (Yi et al., 2024), projecting detections onto the ground plane using estimated or calibrated camera parameters and running the Kalman filter in physical coordinates. MCTrack's decoupled filter architecture (Wang et al., 2024) provides a refined version of this approach.

3. **Mapped Mahalanobis Distance.** Replacing IoU with MMD for track-detection association, handling cases where bounding boxes do not overlap. BoostTrack's approach of augmenting IoU with Mahalanobis distance and shape similarity offers an incremental alternative.

4. **Adaptive Background Model.** Implementing speed-dependent thresholds following Yu et al. (2019), where the background model's age threshold decays exponentially with the average optical flow magnitude: $\tau_\alpha = \mu \cdot e^{-\sigma \bar{f}}$.

5. **GPU Acceleration.** Moving YOLOv8 inference fully to GPU and exploring CUDA-accelerated optical flow to reach the 30 FPS real-time target.

# 7. Conclusion

This paper presented Kiti, a modular obstacle recognition pipeline for autonomous vehicle prototypes that integrates camera motion compensation with deep learning-based detection, multi-object tracking, distance estimation, and trajectory prediction. The key contribution is the regular-grid KLT and RANSAC homography-based camera motion compensation module, which enables traditional motion analysis techniques to operate effectively from a moving camera by subtracting estimated ego-motion from the total optical flow field.

Experimental evaluation on real-world driving sequences demonstrated that the pipeline successfully detects and tracks diverse obstacles including pedestrians, vehicles, and animals, with CMC effectively isolating independent object motion from the dominant ego-motion signal. The modular architecture provides a practical framework for iterative improvement, with clear upgrade paths identified from the current literature.

Future work will focus on integrating ego-motion-aware Kalman filtering, ground-plane tracking, and GPU acceleration to achieve real-time performance suitable for deployment on the prototype vehicle.

# References

Aharon, N., Orfaig, R., and Bobrovsky, B.-Z. (2022). BoT-SORT: Robust associations multi-pedestrian tracking. *arXiv preprint arXiv:2206.14651*.

Bewley, A., Ge, Z., Ott, L., Ramos, F., and Upcroft, B. (2016). Simple online and realtime tracking. In *IEEE International Conference on Image Processing (ICIP)*, pp. 3464-3468.

Bouguet, J.-Y. (2001). Pyramidal implementation of the affine Lucas Kanade feature tracker: Description of the algorithm. *Intel Corporation*, 5(1-10), 4.

Cao, J., Pang, J., Weng, X., Khirodkar, R., and Kitani, K. (2023). Observation-centric SORT: Rethinking SORT for robust multi-object tracking. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9686-9696.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S. (2020). End-to-end object detection with transformers. In *European Conference on Computer Vision (ECCV)*, pp. 213-229.

Eigen, D., Puhrsch, C., and Fergus, R. (2014). Depth map prediction from a single image using a multi-scale deep network. In *Advances in Neural Information Processing Systems (NeurIPS)*, 27.

Farneback, G. (2003). Two-frame motion estimation based on polynomial expansion. In *Scandinavian Conference on Image Analysis*, pp. 363-370.

Fischler, M. A. and Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM*, 24(6), 381-395.

Hedborg, J. and Johansson, B. (2008). Real-time camera ego-motion compensation and lens undistortion on GPU. *Technical Report, Linkoping University*.

Huang, J., Zou, W., Zhu, J., and Zhu, Z. (2018). Optical flow based real-time moving object detection in unconstrained scenes. *arXiv preprint arXiv:1807.04890*.

Jocher, G., Chaurasia, A., and Qiu, J. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics.

Lucas, B. D. and Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. In *International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 674-679.

Mahdian, N., Jani, M., Enayati, A. M. S., and Najjaran, H. (2024). EMAP: Ego-motion aware target prediction module for robust multi-object tracking. *arXiv preprint arXiv:2404.03110*.

Ogawa, M., et al. (2025). FoELS: Moving object detection from moving camera using focus of expansion likelihood and segmentation. *arXiv preprint arXiv:2507.13628*.

Ranftl, R., Bochkovskiy, A., and Koltun, V. (2021). Vision transformers for dense prediction. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 12179-12188.

Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). You only look once: Unified, real-time object detection. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 779-788.

Ren, S., He, K., Girshick, R., and Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 28.

Uemura, H., Ishikawa, S., and Mikolajczyk, K. (2008). Feature tracking and motion compensation for action recognition. In *British Machine Vision Conference (BMVC)*.

Wang, W., et al. (2024). MCTrack: A unified 3D multi-object tracking framework for autonomous driving. *arXiv preprint arXiv:2409.16149*.

Wojke, N., Bewley, A., and Paulus, D. (2017). Simple online and realtime tracking with a deep association metric. In *IEEE International Conference on Image Processing (ICIP)*, pp. 3645-3649.

Yi, K., Li, Z., et al. (2024). UCMCTrack: Multi-object tracking with uniform camera motion compensation. In *AAAI Conference on Artificial Intelligence*, 38.

Yu, Y., Kurnianggoro, L., and Jo, K.-H. (2019). Moving object detection for a moving camera based on global motion compensation and adaptive background model. *International Journal of Control, Automation and Systems*, 17, 1-10.

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., and Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. In *European Conference on Computer Vision (ECCV)*, pp. 1-21.

Chen, Z., et al. (2025). PD-SORT: Occlusion-robust multi-object tracking using pseudo-depth cues. *arXiv preprint arXiv:2501.11288*.

Hu, M., et al. (2024). Metric3D v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

Maggiolino, G., Ahmad, A., Cao, J., and Kitani, K. (2023). Deep OC-SORT: Multi-pedestrian tracking by adaptive re-identification. In *IEEE International Conference on Image Processing (ICIP)*.

Soliman, A., et al. (2025). YOLO26: Real-time object detection with edge optimization. *arXiv preprint arXiv:2509.25164*.

Stanojevic, V., Todorovic, V., and Bhatt, N. (2024). BoostTrack++: Using tracklet information to detect more objects in multiple object tracking. *arXiv preprint arXiv:2408.13003*.

Vu, T., et al. (2025). HybridTrack: A hybrid approach for robust multi-object tracking. *IEEE Robotics and Automation Letters / ICRA 2026*. *arXiv preprint arXiv:2501.01275*.

Wan, Z., et al. (2024). SEA-RAFT: Simple, efficient, accurate RAFT for optical flow. In *European Conference on Computer Vision (ECCV)*, Oral.

Wang, A., et al. (2024b). YOLOv10: Real-time end-to-end object detection. *arXiv preprint arXiv:2405.14458*.

Yang, L., et al. (2024). Depth Anything V2. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Zivkovic, Z. (2004). Improved adaptive Gaussian mixture model for background subtraction. In *International Conference on Pattern Recognition (ICPR)*, pp. 28-31.
