"""
Optical flow module for motion analysis and trajectory prediction.
Implements optical flow calculation, motion vector analysis, and path prediction.
"""
import logging
from typing import List, Tuple, Optional, Dict
from collections import deque
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

from config.settings import settings


logger = logging.getLogger(__name__)


class OpticalFlowAnalyzer:
    """Analyze optical flow for motion detection and trajectory prediction."""
    
    def __init__(self):
        """Initialize optical flow analyzer."""
        self.prev_gray = None
        self.trajectory = deque(maxlen=settings.TRAJECTORY_MAX_LENGTH)
        self.frame_count = 0
        
        # Kalman filter for trajectory prediction
        self.kalman_filter = None
        self.predicted_points = []
        
        logger.info("Initialized OpticalFlowAnalyzer")
    
    def initialize_kalman_filter(self):
        """Initialize Kalman filter for trajectory prediction."""
        # State: [x, y, vx, vy], Measurement: [x, y]
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        
        # Measurement matrix
        self.kalman_filter.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Transition matrix
        self.kalman_filter.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Process noise covariance
        self.kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * settings.KALMAN_PROCESS_NOISE
        
        # Measurement noise covariance
        self.kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * settings.KALMAN_MEASUREMENT_NOISE
        
        logger.info("Kalman filter initialized")
    
    def calculate_optical_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate optical flow between previous and current frame.
        
        Args:
            frame: Current frame (BGR format)
            
        Returns:
            Optional[np.ndarray]: Optical flow field or None if no previous frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Calculate Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            settings.OPTICAL_FLOW_PYR_SCALE,
            settings.OPTICAL_FLOW_LEVELS,
            settings.OPTICAL_FLOW_WINSIZE,
            settings.OPTICAL_FLOW_ITERATIONS,
            settings.OPTICAL_FLOW_POLY_N,
            settings.OPTICAL_FLOW_POLY_SIGMA,
            settings.OPTICAL_FLOW_FLAGS
        )
        
        self.prev_gray = gray
        return flow
    
    def detect_motion(self, flow: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect motion from optical flow field.
        
        Args:
            flow: Optical flow field
            
        Returns:
            Tuple[np.ndarray, List]: Motion mask and contours
        """
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create motion mask
        motion_mask = (mag > settings.MOTION_THRESHOLD).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        return motion_mask, contours
    
    def find_largest_moving_object(self, contours: List) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the largest moving object from contours.
        
        Args:
            contours: List of contours
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Bounding box (x, y, w, h) or None
        """
        largest = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > settings.MIN_CONTOUR_AREA and area > max_area:
                max_area = area
                largest = cnt
        
        if largest is not None:
            return cv2.boundingRect(largest)
        
        return None
    
    def update_trajectory(self, bbox: Tuple[int, int, int, int]):
        """
        Update trajectory with new position.
        
        Args:
            bbox: Bounding box (x, y, w, h)
        """
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        
        self.trajectory.append((self.frame_count, cx, cy))
        self.frame_count += 1
    
    def predict_linear_trajectory(self, num_frames: int = None) -> List[Tuple[int, int]]:
        """
        Predict future trajectory using linear regression.
        
        Args:
            num_frames: Number of frames to predict (uses default if None)
            
        Returns:
            List[Tuple[int, int]]: Predicted (x, y) positions
        """
        if len(self.trajectory) < settings.MIN_TRAJECTORY_LENGTH:
            return []
        
        num_frames = num_frames or settings.PREDICTION_FRAMES
        
        # Convert trajectory to numpy array
        arr = np.array(self.trajectory)
        times = arr[:, 0].reshape(-1, 1)
        xs = arr[:, 1]
        ys = arr[:, 2]
        
        # Fit linear regression models
        model_x = LinearRegression().fit(times, xs)
        model_y = LinearRegression().fit(times, ys)
        
        # Predict future positions
        future_times = np.array([[self.frame_count + i] for i in range(1, num_frames + 1)])
        pred_xs = model_x.predict(future_times).astype(int)
        pred_ys = model_y.predict(future_times).astype(int)
        
        predictions = list(zip(pred_xs, pred_ys))
        return predictions
    
    def predict_kalman_trajectory(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Predict next position using Kalman filter.
        
        Args:
            bbox: Current bounding box (x, y, w, h)
            
        Returns:
            Tuple[int, int]: Predicted (x, y) position
        """
        if self.kalman_filter is None:
            self.initialize_kalman_filter()
        
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        
        # Update Kalman filter with measurement
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman_filter.correct(measurement)
        
        # Predict next state
        prediction = self.kalman_filter.predict()
        px = int(prediction[0])
        py = int(prediction[1])
        
        self.predicted_points.append((px, py))
        
        return px, py
    
    def visualize_flow(self, frame: np.ndarray, flow: np.ndarray, 
                      step: int = 16) -> np.ndarray:
        """
        Visualize optical flow as vector field.
        
        Args:
            frame: Input frame
            flow: Optical flow field
            step: Sampling step for flow vectors
            
        Returns:
            np.ndarray: Frame with flow vectors drawn
        """
        h, w = frame.shape[:2]
        vis_frame = frame.copy()
        
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        
        # Create line endpoints
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        # Draw flow vectors
        for (x1, y1), (x2, y2) in lines:
            cv2.arrowedLine(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1, 
                          cv2.LINE_AA, tipLength=settings.ARROW_TIP_LENGTH)
        
        return vis_frame
    
    def process_frame_with_prediction(self, frame: np.ndarray,
                                     draw_trajectory: bool = True,
                                     draw_prediction: bool = True,
                                     use_kalman: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Process frame with optical flow and trajectory prediction.
        
        Args:
            frame: Input frame (BGR format)
            draw_trajectory: Whether to draw trajectory
            draw_prediction: Whether to draw predicted path
            use_kalman: Whether to use Kalman filter (else linear regression)
            
        Returns:
            Tuple[np.ndarray, Dict]: Processed frame and analysis results
        """
        result_frame = frame.copy()
        results = {
            'motion_detected': False,
            'current_position': None,
            'predicted_position': None,
            'trajectory_length': len(self.trajectory)
        }
        
        # Calculate optical flow
        flow = self.calculate_optical_flow(frame)
        
        if flow is None:
            return result_frame, results
        
        # Detect motion
        motion_mask, contours = self.detect_motion(flow)
        
        # Find largest moving object
        bbox = self.find_largest_moving_object(contours)
        
        if bbox is not None:
            x, y, w, h = bbox
            cx = x + w // 2
            cy = y + h // 2
            
            results['motion_detected'] = True
            results['current_position'] = (cx, cy)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(result_frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Update trajectory
            self.update_trajectory(bbox)
            
            # Draw trajectory
            if draw_trajectory and len(self.trajectory) > 1:
                points = [(int(t[1]), int(t[2])) for t in self.trajectory]
                for i in range(1, len(points)):
                    cv2.line(result_frame, points[i-1], points[i], (255, 255, 0), 2)
            
            # Predict and draw future path
            if draw_prediction:
                if use_kalman:
                    px, py = self.predict_kalman_trajectory(bbox)
                    results['predicted_position'] = (px, py)
                    cv2.circle(result_frame, (px, py), 5, (255, 0, 0), -1)
                    
                    # Draw Kalman trajectory
                    if len(self.predicted_points) > 1:
                        for i in range(1, len(self.predicted_points)):
                            cv2.line(result_frame, self.predicted_points[i-1],
                                   self.predicted_points[i], (255, 0, 0), 2)
                else:
                    predictions = self.predict_linear_trajectory()
                    for px, py in predictions:
                        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                            cv2.circle(result_frame, (px, py), 2, (255, 0, 0), -1)
        
        return result_frame, results
    
    def reset(self):
        """Reset analyzer state."""
        self.prev_gray = None
        self.trajectory.clear()
        self.predicted_points.clear()
        self.frame_count = 0
        self.kalman_filter = None
        logger.info("Optical flow analyzer reset")
