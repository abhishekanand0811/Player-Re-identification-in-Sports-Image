"""Advanced features for improved player re-identification"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    """Advanced feature extraction methods for better re-identification"""
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=100)
        self.sift = cv2.SIFT_create(nfeatures=50)
        
    def extract_keypoint_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract ORB keypoint features"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is not None:
            # Use mean of descriptors as feature vector
            return np.mean(descriptors, axis=0)
        else:
            return np.zeros(32)  # ORB descriptor size
    
    def extract_texture_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Patterns"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Simple LBP implementation
        def lbp(image, radius=1, n_points=8):
            h, w = image.shape
            lbp_image = np.zeros((h, w), dtype=np.uint8)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = image[i, j]
                    binary_string = ''
                    
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        
                        if image[x, y] >= center:
                            binary_string += '1'
                        else:
                            binary_string += '0'
                    
                    lbp_image[i, j] = int(binary_string, 2)
            
            return lbp_image
        
        lbp_image = lbp(gray)
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        return hist / (hist.sum() + 1e-7)
    
    def extract_edge_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge direction histogram
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        # Histogram of oriented gradients (simplified)
        hist, _ = np.histogram(direction[magnitude > 10], bins=18, range=(-np.pi, np.pi))
        return hist / (hist.sum() + 1e-7)

class MotionAnalyzer:
    """Analyze player motion patterns for better tracking"""
    
    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.position_history: Dict[int, List[Tuple[float, float]]] = {}
        
    def update_position(self, player_id: int, center: Tuple[float, float]):
        """Update position history for a player"""
        if player_id not in self.position_history:
            self.position_history[player_id] = []
        
        self.position_history[player_id].append(center)
        
        # Keep only recent history
        if len(self.position_history[player_id]) > self.history_length:
            self.position_history[player_id] = self.position_history[player_id][-self.history_length:]
    
    def predict_next_position(self, player_id: int) -> Tuple[float, float]:
        """Predict next position based on motion history"""
        if player_id not in self.position_history or len(self.position_history[player_id]) < 2:
            return None
        
        positions = self.position_history[player_id]
        
        # Simple linear prediction
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
            
            next_x = positions[-1][0] + dx
            next_y = positions[-1][1] + dy
            
            return (next_x, next_y)
        
        return positions[-1]
    
    def calculate_motion_similarity(self, player_id1: int, player_id2: int) -> float:
        """Calculate similarity based on motion patterns"""
        if (player_id1 not in self.position_history or 
            player_id2 not in self.position_history):
            return 0.0
        
        pos1 = self.position_history[player_id1]
        pos2 = self.position_history[player_id2]
        
        if len(pos1) < 3 or len(pos2) < 3:
            return 0.0
        
        # Calculate velocity vectors
        vel1 = [(pos1[i][0] - pos1[i-1][0], pos1[i][1] - pos1[i-1][1]) 
                for i in range(1, len(pos1))]
        vel2 = [(pos2[i][0] - pos2[i-1][0], pos2[i][1] - pos2[i-1][1]) 
                for i in range(1, len(pos2))]
        
        # Compare recent velocities
        if vel1 and vel2:
            recent_vel1 = vel1[-min(3, len(vel1)):]
            recent_vel2 = vel2[-min(3, len(vel2)):]
            
            similarities = []
            for v1 in recent_vel1:
                for v2 in recent_vel2:
                    # Cosine similarity of velocity vectors
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
                    norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if norm1 > 0 and norm2 > 0:
                        sim = dot_product / (norm1 * norm2)
                        similarities.append(max(0, sim))
            
            return np.mean(similarities) if similarities else 0.0
        
        return 0.0

class AdaptiveThresholdManager:
    """Dynamically adjust thresholds based on tracking performance"""
    
    def __init__(self, initial_threshold: float = 0.3):
        self.threshold = initial_threshold
        self.performance_history = []
        self.adjustment_factor = 0.05
        
    def update_performance(self, successful_matches: int, total_detections: int):
        """Update performance metrics"""
        if total_detections > 0:
            match_rate = successful_matches / total_detections
            self.performance_history.append(match_rate)
            
            # Keep only recent history
            if len(self.performance_history) > 20:
                self.performance_history = self.performance_history[-20:]
    
    def adjust_threshold(self):
        """Adjust threshold based on recent performance"""
        if len(self.performance_history) < 5:
            return
        
        recent_performance = np.mean(self.performance_history[-5:])
        
        # If performance is too low, lower threshold (more lenient matching)
        if recent_performance < 0.3:
            self.threshold = max(0.1, self.threshold - self.adjustment_factor)
            logger.info(f"Lowered similarity threshold to {self.threshold:.3f}")
        
        # If performance is very high, raise threshold (stricter matching)
        elif recent_performance > 0.8:
            self.threshold = min(0.6, self.threshold + self.adjustment_factor)
            logger.info(f"Raised similarity threshold to {self.threshold:.3f}")
    
    def get_threshold(self) -> float:
        """Get current threshold"""
        return self.threshold
