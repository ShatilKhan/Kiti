"""
Kiti Unified Obstacle Recognition Pipeline
============================================
Combined pipeline merging Area Marking + Optical Flow into a single
modular obstacle recognition system for autonomous vehicle prototype.

Pipeline Stages:
    1. Video I/O — Load videos from videos/ directory
    2. Camera Motion Compensation — Homography-based frame alignment (KLT + RANSAC)
    3. Dense Optical Flow — Farneback with ego-motion residual
    4. Background Subtraction — MOG2 on compensated frames
    5. Object Detection — YOLOv8
    6. Object Tracking — SORT with Kalman filter
    7. Distance Estimation — Focal-length based monocular depth
    8. Trajectory Prediction — Kalman filter + linear regression
    9. Region of Interest — Central region marking with behavior detection
    10. Annotation & Logging — Overlay results, export CSV/JSON

Usage:
    python kiti_pipeline.py                          # Process first video
    python kiti_pipeline.py --video 2                # Process 3rd video (0-indexed)
    python kiti_pipeline.py --max-frames 300         # Limit frames for testing
    python kiti_pipeline.py --no-cmc                 # Disable camera motion compensation
    python kiti_pipeline.py --batch                  # Process all videos
"""

import os
import sys
import glob
import math
import json
import time
import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# YOLO
from ultralytics import YOLO

# SORT tracker
sys.path.insert(0, '/tmp/sort')
try:
    from sort import Sort
except ImportError:
    print("SORT not found. Installing...")
    os.system('pip install filterpy scikit-image --quiet')
    if not os.path.exists('/tmp/sort'):
        os.system('git clone https://github.com/abewley/sort.git /tmp/sort')
    sort_py = Path('/tmp/sort/sort.py')
    sort_py.write_text(
        sort_py.read_text().replace("matplotlib.use('TkAgg')", "matplotlib.use('Agg')")
    )
    from sort import Sort


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Paths
    'video_dir': 'videos',
    'output_dir': 'output',

    # Detection
    'yolo_model': 'models/yolov8n.pt',
    'yolo_conf': 0.3,

    # Region of Interest (central percentage of frame width)
    'roi_width_pct': 0.40,

    # Distance estimation
    'focal_length': 353,
    'real_object_height': 1.7,  # meters (average person)

    # Optical flow (Farneback)
    'flow_pyr_scale': 0.5,
    'flow_levels': 3,
    'flow_winsize': 15,
    'flow_iterations': 3,
    'flow_poly_n': 5,
    'flow_poly_sigma': 1.2,
    'motion_threshold': 1.0,

    # Background subtraction (MOG2)
    'mog2_history': 500,
    'mog2_var_threshold': 50,

    # Camera Motion Compensation
    'cmc_enabled': True,
    'cmc_grid_size': 16,
    'cmc_ransac_threshold': 3.0,
    'cmc_min_inliers': 10,

    # SORT tracker
    'sort_max_age': 5,
    'sort_min_hits': 3,
    'sort_iou_threshold': 0.3,

    # Trajectory prediction
    'trajectory_history': 30,
    'prediction_horizon': 30,

    # Behavior detection
    'movement_threshold': 10,

    # Processing
    'max_frames': None,
}


# =============================================================================
# Camera Motion Compensation
# =============================================================================

class CameraMotionCompensator:
    """Estimates and compensates for camera ego-motion between frames.

    Uses regular grid KLT tracking + RANSAC homography to align frames.
    Regular grid avoids bias toward high-contrast moving objects.

    References:
        - Hedborg & Johansson: GPU Ego-Motion Compensation
        - Yu et al. (IJCAS 2019): Moving Object Detection for Moving Camera
        - Uemura et al. (BMVC 2008): Feature Tracking & Dominant Plane
    """

    def __init__(self, grid_size=16, ransac_thresh=3.0, min_inliers=10):
        self.grid_size = grid_size
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.prev_gray = None
        self.prev_pts = None
        self.last_homography = np.eye(3, dtype=np.float64)

        # Pyramidal Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def _generate_grid_points(self, h, w):
        """Generate regular grid of feature points.

        Using a regular grid instead of Harris/FAST corners avoids
        the problem of feature points clustering on moving objects.
        """
        ys = np.arange(self.grid_size // 2, h, self.grid_size)
        xs = np.arange(self.grid_size // 2, w, self.grid_size)
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 1, 2).astype(np.float32)
        return grid

    def estimate_homography(self, gray_frame):
        """Estimate homography from previous frame to current frame.

        Returns:
            H: 3x3 homography matrix (identity if first frame or failure)
            inlier_ratio: fraction of points consistent with homography
        """
        h, w = gray_frame.shape[:2]

        if self.prev_gray is None:
            self.prev_gray = gray_frame.copy()
            self.prev_pts = self._generate_grid_points(h, w)
            return np.eye(3, dtype=np.float64), 1.0

        # Track grid points with KLT
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_frame, self.prev_pts, None, **self.lk_params
        )

        status = status.flatten()
        good_prev = self.prev_pts[status == 1].reshape(-1, 2)
        good_curr = curr_pts[status == 1].reshape(-1, 2)

        if len(good_prev) < self.min_inliers:
            self.prev_gray = gray_frame.copy()
            self.prev_pts = self._generate_grid_points(h, w)
            return self.last_homography.copy(), 0.0

        # RANSAC homography
        H, mask = cv2.findHomography(
            good_prev, good_curr, cv2.RANSAC, self.ransac_thresh
        )

        if H is None:
            self.prev_gray = gray_frame.copy()
            self.prev_pts = self._generate_grid_points(h, w)
            return self.last_homography.copy(), 0.0

        inlier_ratio = float(np.sum(mask)) / len(mask) if mask is not None else 0.0
        self.last_homography = H

        self.prev_gray = gray_frame.copy()
        self.prev_pts = self._generate_grid_points(h, w)

        return H, inlier_ratio

    def warp_frame(self, frame, H):
        """Warp frame using homography for alignment."""
        h, w = frame.shape[:2]
        return cv2.warpPerspective(
            frame, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    def compute_residual_flow(self, flow, H, h, w):
        """Subtract ego-motion from optical flow to isolate object motion.

        The homography H describes background point motion between frames.
        Subtracting this from total flow isolates independently moving objects.
        """
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        ones = np.ones_like(xs)
        pts = np.stack([xs, ys, ones], axis=-1)

        H_float = H.astype(np.float32)
        transformed = np.einsum('ij,hwj->hwi', H_float, pts)

        z = transformed[..., 2:3]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        transformed_xy = transformed[..., :2] / z

        ego_flow = transformed_xy - np.stack([xs, ys], axis=-1)
        residual = flow - ego_flow

        return residual


# =============================================================================
# Optical Flow Analysis
# =============================================================================

class OpticalFlowAnalyzer:
    """Dense optical flow with ego-motion compensation."""

    def __init__(self, config):
        self.pyr_scale = config['flow_pyr_scale']
        self.levels = config['flow_levels']
        self.winsize = config['flow_winsize']
        self.iterations = config['flow_iterations']
        self.poly_n = config['flow_poly_n']
        self.poly_sigma = config['flow_poly_sigma']
        self.motion_threshold = config['motion_threshold']
        self.prev_gray = None

    def compute_flow(self, gray):
        """Compute dense Farneback optical flow."""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            h, w = gray.shape[:2]
            return np.zeros((h, w, 2), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, 0
        )
        self.prev_gray = gray.copy()
        return flow

    def flow_to_motion_mask(self, flow, threshold=None):
        """Convert flow to binary motion mask via magnitude threshold."""
        if threshold is None:
            threshold = self.motion_threshold
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return (mag > threshold).astype(np.uint8) * 255

    def flow_to_hsv(self, flow):
        """Visualize optical flow as HSV color image."""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# =============================================================================
# Trajectory Prediction
# =============================================================================

class TrajectoryPredictor:
    """Kalman filter + linear regression trajectory prediction."""

    def __init__(self, history_len=30, prediction_horizon=30):
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        self.trajectories = defaultdict(lambda: deque(maxlen=history_len))
        self.kalman_filters = {}

    def _create_kalman(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        return kf

    def update(self, track_id, cx, cy, frame_id):
        """Update trajectory and return Kalman prediction."""
        self.trajectories[track_id].append((frame_id, cx, cy))

        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self._create_kalman()

        kf = self.kalman_filters[track_id]
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        kf.correct(measurement)
        prediction = kf.predict()

        return int(prediction[0].item()), int(prediction[1].item())

    def predict_path(self, track_id, frame_id):
        """Predict future path using linear regression."""
        traj = self.trajectories.get(track_id)
        if traj is None or len(traj) < 10:
            return []

        arr = np.array(list(traj))
        times = arr[:, 0].reshape(-1, 1)

        model_x = LinearRegression().fit(times, arr[:, 1])
        model_y = LinearRegression().fit(times, arr[:, 2])

        future = np.array([[frame_id + i] for i in range(1, self.prediction_horizon + 1)])
        pred_xs = model_x.predict(future).astype(int)
        pred_ys = model_y.predict(future).astype(int)

        return list(zip(pred_xs.tolist(), pred_ys.tolist()))

    def cleanup(self, active_ids):
        stale = set(self.trajectories.keys()) - set(active_ids)
        for tid in stale:
            del self.trajectories[tid]
            self.kalman_filters.pop(tid, None)


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_distance(bbox_height, focal_length, real_height):
    """Monocular distance: d = (real_height * focal_length) / pixel_height"""
    if bbox_height < 10:
        return None
    return round((real_height * focal_length) / bbox_height, 2)


def angle_to_direction(angle):
    """Convert angle to 8-direction compass."""
    directions = ['East', 'NE', 'North', 'NW', 'West', 'SW', 'South', 'SE']
    return directions[int(((angle + 22.5) % 360) // 45)]


def get_movement_direction(prev_x, prev_y, curr_x, curr_y):
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    angle = math.degrees(math.atan2(-dy, dx))
    angle = (angle + 360) % 360
    return angle, angle_to_direction(angle)


def describe_behavior(prev_pos, curr_x, curr_y, name, threshold=10):
    if prev_pos is None:
        return f"{name} entered", None

    prev_x, prev_y = prev_pos
    angle, direction = get_movement_direction(prev_x, prev_y, curr_x, curr_y)
    dx = abs(curr_x - prev_x)
    dy = abs(curr_y - prev_y)

    if dx < threshold and dy < threshold:
        return f"{name} steady", direction
    return f"{name} moving {direction}", direction


def list_videos(video_dir):
    videos = []
    for ext in ['*.mp4', '*.mkv', '*.avi', '*.mov']:
        videos.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(videos)


# =============================================================================
# Main Pipeline
# =============================================================================

def process_video(video_path, config):
    """Process a single video through the full Kiti pipeline."""
    video_name = Path(video_path).stem
    print(f"\n{'=' * 60}")
    print(f"Processing: {video_name}")
    print(f"{'=' * 60}")

    # Initialize components
    model = YOLO(config['yolo_model'])
    tracker = Sort(
        max_age=config['sort_max_age'],
        min_hits=config['sort_min_hits'],
        iou_threshold=config['sort_iou_threshold']
    )
    cmc = CameraMotionCompensator(
        grid_size=config['cmc_grid_size'],
        ransac_thresh=config['cmc_ransac_threshold'],
        min_inliers=config['cmc_min_inliers']
    )
    flow_analyzer = OpticalFlowAnalyzer(config)
    traj_predictor = TrajectoryPredictor(
        history_len=config['trajectory_history'],
        prediction_horizon=config['prediction_horizon']
    )
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=config['mog2_history'],
        varThreshold=config['mog2_var_threshold']
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return None

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  {fw}x{fh} @ {fps}fps, {total_frames} frames")

    # Output setup
    msg_bar_h = 60
    out_h = fh + msg_bar_h
    roi_margin = (1.0 - config['roi_width_pct']) / 2
    roi_left = int(roi_margin * fw)
    roi_right = int((1.0 - roi_margin) * fw)

    os.makedirs(config['output_dir'], exist_ok=True)
    out_ann_path = os.path.join(config['output_dir'], f'{video_name}_annotated.mp4')
    out_flow_path = os.path.join(config['output_dir'], f'{video_name}_flow.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_ann = cv2.VideoWriter(out_ann_path, fourcc, fps, (fw, out_h))
    out_flow = cv2.VideoWriter(out_flow_path, fourcc, fps, (fw, fh))

    # State
    prev_centers = {}
    logs = []
    motion_heatmap = np.zeros((fh, fw), dtype=np.float32)
    frame_id = 0
    processing_times = []
    limit = config['max_frames'] or total_frames

    print(f"  Processing up to {min(limit, total_frames)} frames...")

    while cap.isOpened() and frame_id < limit:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()
        frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Stage 1: Camera Motion Compensation ----
        if config['cmc_enabled']:
            H, inlier_ratio = cmc.estimate_homography(gray)
        else:
            H, inlier_ratio = np.eye(3, dtype=np.float64), 1.0

        # ---- Stage 2: Dense Optical Flow ----
        flow = flow_analyzer.compute_flow(gray)

        if config['cmc_enabled'] and not np.allclose(H, np.eye(3)):
            residual_flow = cmc.compute_residual_flow(flow, H, fh, fw)
        else:
            residual_flow = flow

        motion_mask = flow_analyzer.flow_to_motion_mask(residual_flow)
        motion_heatmap += (motion_mask / 255).astype(np.float32)

        # Flow visualization
        flow_vis = flow_analyzer.flow_to_hsv(residual_flow)
        out_flow.write(flow_vis)

        # ---- Stage 3: Background Subtraction ----
        if config['cmc_enabled'] and not np.allclose(H, np.eye(3)):
            compensated = cmc.warp_frame(frame, np.linalg.inv(H))
            fgmask = fgbg.apply(compensated)
        else:
            fgmask = fgbg.apply(frame)

        _, fg_thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fg_thresh = cv2.morphologyEx(fg_thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        combined_mask = cv2.bitwise_or(fg_thresh, motion_mask)

        # ---- Stage 4: Object Detection ----
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        dets = []
        det_classes = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                dets.append([x1, y1, x2, y2, conf])
                det_classes.append(cls)

        # ---- Stage 5: Object Tracking ----
        if len(dets) > 0:
            tracked = tracker.update(np.array(dets))
        else:
            tracked = tracker.update(np.empty((0, 5)))

        # ---- Stage 6: Annotation ----
        annotated = np.zeros((out_h, fw, 3), dtype=np.uint8)
        annotated[:fh] = frame.copy()

        # Draw ROI
        overlay = annotated.copy()
        cv2.rectangle(overlay, (roi_left, 0), (roi_right, fh), (0, 255, 0), 2)
        annotated = overlay

        messages = []
        active_ids = []

        for obj in tracked:
            x1, y1, x2, y2, track_id = map(int, obj)
            active_ids.append(track_id)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bbox_h = y2 - y1

            # Match to YOLO detection for class name
            class_name = "object"
            matched_conf = None
            for det, cls_idx in zip(dets, det_classes):
                if abs(det[0] - x1) < 30 and abs(det[1] - y1) < 30:
                    class_name = model.names[cls_idx]
                    matched_conf = det[4]
                    break

            # Distance
            distance = estimate_distance(
                bbox_h, config['focal_length'], config['real_object_height']
            )

            # Trajectory
            pred_x, pred_y = traj_predictor.update(track_id, cx, cy, frame_id)
            future_path = traj_predictor.predict_path(track_id, frame_id)

            # ROI check
            in_roi = roi_left <= cx <= roi_right

            # Behavior
            prev_pos = prev_centers.get(track_id)
            behavior, direction = describe_behavior(
                prev_pos, cx, cy, class_name, config['movement_threshold']
            )
            prev_centers[track_id] = (cx, cy)

            # Draw
            color = (0, 0, 255) if in_roi else (255, 180, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} ID:{track_id}"
            if distance is not None:
                label += f" {distance}m"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Kalman prediction marker
            cv2.circle(annotated, (pred_x, pred_y), 4, (255, 0, 255), -1)

            # Future path
            for px, py in future_path:
                if 0 <= px < fw and 0 <= py < fh:
                    cv2.circle(annotated, (px, py), 2, (255, 255, 0), -1)

            if in_roi:
                msg = f"ID:{track_id} ({class_name}) {behavior}"
                if distance is not None:
                    msg += f" [{distance}m]"
                messages.append(msg)

            logs.append({
                'frame': frame_id,
                'track_id': track_id,
                'class': class_name,
                'confidence': matched_conf,
                'bbox': [x1, y1, x2, y2],
                'center': [cx, cy],
                'distance_m': distance,
                'in_roi': in_roi,
                'behavior': behavior,
                'direction': direction,
                'kalman_pred': [pred_x, pred_y],
                'cmc_inlier_ratio': round(inlier_ratio, 3),
            })

        traj_predictor.cleanup(active_ids)

        # Message bar
        cv2.rectangle(annotated, (0, fh), (fw, out_h), (0, 200, 200), -1)
        for idx, msg in enumerate(messages[:2]):
            cv2.putText(annotated, msg, (10, fh + 25 + idx * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Status indicators
        cmc_text = f"CMC: {inlier_ratio:.0%}" if config['cmc_enabled'] else "CMC: OFF"
        cv2.putText(annotated, cmc_text, (fw - 130, fh + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(annotated, f"F:{frame_id}/{total_frames}",
                    (fw - 130, fh + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        out_ann.write(annotated)

        t_elapsed = time.time() - t_start
        processing_times.append(t_elapsed)

        if frame_id % 100 == 0:
            avg_fps = 1.0 / np.mean(processing_times[-100:])
            pct = frame_id / total_frames * 100
            print(f"  Frame {frame_id}/{total_frames} ({pct:.0f}%) - {avg_fps:.1f} fps")

    cap.release()
    out_ann.release()
    out_flow.release()

    total_time = sum(processing_times)
    avg_fps = frame_id / total_time if total_time > 0 else 0

    print(f"\n  Done: {frame_id} frames in {total_time:.1f}s ({avg_fps:.1f} avg fps)")
    print(f"  Annotated: {out_ann_path}")
    print(f"  Flow:      {out_flow_path}")

    # ---- Save logs ----
    try:
        import pandas as pd
        csv_path = os.path.join(config['output_dir'], f'{video_name}_detections.csv')
        pd.DataFrame(logs).to_csv(csv_path, index=False)
        print(f"  CSV log:   {csv_path} ({len(logs)} entries)")
    except ImportError:
        pass

    json_path = os.path.join(config['output_dir'], f'{video_name}_detections.json')
    with open(json_path, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"  JSON log:  {json_path}")

    # ---- Motion heatmap ----
    norm_heatmap = cv2.normalize(motion_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    colored_heatmap = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_path = os.path.join(config['output_dir'], f'{video_name}_heatmap.png')
    cv2.imwrite(heatmap_path, colored_heatmap)
    print(f"  Heatmap:   {heatmap_path}")

    # ---- Performance plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Motion Heatmap (Accumulated)')
    axes[0].axis('off')

    axes[1].plot(processing_times, alpha=0.3, color='blue')
    if len(processing_times) > 10:
        rolling = np.convolve(processing_times, np.ones(10) / 10, mode='valid')
        axes[1].plot(range(9, 9 + len(rolling)), rolling, color='red', label='10-frame avg')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Processing Time (s)')
    axes[1].set_title(f'Performance ({avg_fps:.1f} avg fps)')
    axes[1].legend()

    plt.tight_layout()
    perf_path = os.path.join(config['output_dir'], f'{video_name}_performance.png')
    plt.savefig(perf_path, dpi=150)
    plt.close()
    print(f"  Perf plot: {perf_path}")

    # Summary
    if logs:
        unique_tracks = len(set(l['track_id'] for l in logs))
        classes = list(set(l['class'] for l in logs))
        roi_count = sum(1 for l in logs if l['in_roi'])
        print(f"\n  Summary:")
        print(f"    Unique tracks: {unique_tracks}")
        print(f"    Classes: {classes}")
        print(f"    ROI detections: {roi_count} ({roi_count / len(logs) * 100:.0f}%)")

    return {
        'video': video_name,
        'frames': frame_id,
        'fps': avg_fps,
        'detections': len(logs),
        'output': out_ann_path,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kiti Obstacle Recognition Pipeline')
    parser.add_argument('--video', type=int, default=0,
                        help='Video index to process (default: 0)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max frames to process (default: all)')
    parser.add_argument('--no-cmc', action='store_true',
                        help='Disable camera motion compensation')
    parser.add_argument('--batch', action='store_true',
                        help='Process all videos')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--video-dir', type=str, default='videos',
                        help='Video directory')
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['max_frames'] = args.max_frames
    config['cmc_enabled'] = not args.no_cmc
    config['output_dir'] = args.output_dir
    config['video_dir'] = args.video_dir

    videos = list_videos(config['video_dir'])
    if not videos:
        print(f"No videos found in {config['video_dir']}/")
        sys.exit(1)

    print(f"Found {len(videos)} videos:")
    for i, v in enumerate(videos):
        print(f"  [{i}] {os.path.basename(v)}")

    if args.batch:
        results = []
        for vp in videos:
            result = process_video(vp, config)
            if result:
                results.append(result)
        print(f"\n{'=' * 60}")
        print("Batch Summary:")
        for r in results:
            print(f"  {r['video']}: {r['frames']}f @ {r['fps']:.1f}fps, {r['detections']} dets")
    else:
        if args.video >= len(videos):
            print(f"Video index {args.video} out of range (0-{len(videos) - 1})")
            sys.exit(1)
        process_video(videos[args.video], config)


if __name__ == '__main__':
    main()
