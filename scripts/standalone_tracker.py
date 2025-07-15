"""Standalone player re-identification script - no external imports needed"""

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Represents a player detection with features"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frame_id: int
    features: np.ndarray
    center: Tuple[float, float]
    
@dataclass
class Player:
    """Represents a tracked player"""
    player_id: int
    detections: List[Detection]
    feature_history: List[np.ndarray]
    last_seen_frame: int
    total_appearances: int
    is_active: bool = True
    
class FeatureExtractor:
    """Extracts visual and spatial features from player detections"""
    
    def __init__(self):
        self.color_bins = 32
        
    def extract_color_histogram(self, image_patch: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        try:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [self.color_bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [self.color_bins], [0, 256])
            
            # Normalize and concatenate
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
            
            return np.concatenate([hist_h, hist_s, hist_v])
        except:
            return np.zeros(self.color_bins * 3)
    
    def extract_spatial_features(self, bbox: Tuple[int, int, int, int], 
                               frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract spatial position features"""
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # Normalize coordinates
        center_x = (x1 + x2) / (2 * width)
        center_y = (y1 + y2) / (2 * height)
        bbox_width = (x2 - x1) / width
        bbox_height = (y2 - y1) / height
        aspect_ratio = bbox_width / (bbox_height + 1e-7)
        
        return np.array([center_x, center_y, bbox_width, bbox_height, aspect_ratio])
    
    def extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract combined features from detection"""
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract image patch
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
            return np.zeros(self.color_bins * 3 + 5)
        
        try:
            # Resize patch for consistency
            patch = cv2.resize(patch, (64, 128))
            
            # Extract features
            color_features = self.extract_color_histogram(patch)
            spatial_features = self.extract_spatial_features(bbox, frame.shape)
            
            return np.concatenate([color_features, spatial_features])
        except:
            return np.zeros(self.color_bins * 3 + 5)

class PlayerTracker:
    """Main player tracking and re-identification system"""
    
    def __init__(self, model_path: str = "yolo11n.pt", 
                 max_disappeared_frames: int = 30,
                 similarity_threshold: float = 0.3):
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
        self.feature_extractor = FeatureExtractor()
        self.players: Dict[int, Player] = {}
        self.next_player_id = 1
        self.max_disappeared_frames = max_disappeared_frames
        self.similarity_threshold = similarity_threshold
        self.frame_count = 0
        
    def detect_players(self, frame: np.ndarray) -> List[Detection]:
        """Detect players in frame using YOLO"""
        try:
            results = self.model(frame, classes=[0], verbose=False)  # class 0 is person
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bbox coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Filter by confidence and reasonable size
                        if confidence > 0.5 and (x2 - x1) > 20 and (y2 - y1) > 40:
                            # Extract features
                            features = self.feature_extractor.extract_features(frame, (x1, y1, x2, y2))
                            center = ((x1 + x2) / 2, (y1 + y2) / 2)
                            
                            detection = Detection(
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                frame_id=self.frame_count,
                                features=features,
                                center=center
                            )
                            detections.append(detection)
            
            return detections
        except Exception as e:
            logger.warning(f"Detection failed for frame {self.frame_count}: {e}")
            return []
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors"""
        try:
            if len(features1) != len(features2) or len(features1) == 0:
                return 0.0
                
            # Use cosine similarity for color features and euclidean for spatial
            color_sim = 1 - cosine(features1[:-5], features2[:-5])
            spatial_sim = 1 / (1 + np.linalg.norm(features1[-5:] - features2[-5:]))
            
            # Handle NaN values
            if np.isnan(color_sim):
                color_sim = 0.0
            if np.isnan(spatial_sim):
                spatial_sim = 0.0
            
            # Weighted combination
            return 0.7 * color_sim + 0.3 * spatial_sim
        except:
            return 0.0
    
    def match_detections_to_players(self, detections: List[Detection]) -> Dict[int, int]:
        """Match current detections to existing players using Hungarian algorithm"""
        if not self.players or not detections:
            return {}
        
        active_players = [p for p in self.players.values() if p.is_active]
        if not active_players:
            return {}
        
        try:
            # Create cost matrix
            cost_matrix = np.zeros((len(active_players), len(detections)))
            
            for i, player in enumerate(active_players):
                # Use average of recent features for matching
                recent_features = player.feature_history[-5:]  # Last 5 frames
                if recent_features:
                    avg_features = np.mean(recent_features, axis=0)
                    
                    for j, detection in enumerate(detections):
                        similarity = self.calculate_similarity(avg_features, detection.features)
                        cost_matrix[i, j] = 1 - similarity  # Convert to cost
            
            # Apply Hungarian algorithm
            player_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
            # Filter matches by threshold
            matches = {}
            for p_idx, d_idx in zip(player_indices, detection_indices):
                if cost_matrix[p_idx, d_idx] < (1 - self.similarity_threshold):
                    player_id = active_players[p_idx].player_id
                    matches[player_id] = d_idx
            
            return matches
        except Exception as e:
            logger.warning(f"Matching failed: {e}")
            return {}
    
    def attempt_reidentification(self, detection: Detection) -> Optional[int]:
        """Attempt to re-identify a detection with inactive players"""
        inactive_players = [p for p in self.players.values() if not p.is_active]
        
        best_match_id = None
        best_similarity = 0
        
        for player in inactive_players:
            if not player.feature_history:
                continue
                
            # Use multiple recent features for better matching
            recent_features = player.feature_history[-10:]  # Last 10 features
            
            similarities = []
            for features in recent_features:
                sim = self.calculate_similarity(features, detection.features)
                similarities.append(sim)
            
            if similarities:
                # Use maximum similarity (best match among recent appearances)
                max_similarity = max(similarities)
                
                if max_similarity > best_similarity and max_similarity > (self.similarity_threshold + 0.1):
                    best_similarity = max_similarity
                    best_match_id = player.player_id
        
        return best_match_id
    
    def update_players(self, detections: List[Detection]):
        """Update player tracks with new detections"""
        matches = self.match_detections_to_players(detections)
        matched_detection_indices = set(matches.values())
        
        # Update matched players
        for player_id, detection_idx in matches.items():
            detection = detections[detection_idx]
            player = self.players[player_id]
            
            player.detections.append(detection)
            player.feature_history.append(detection.features)
            player.last_seen_frame = self.frame_count
            player.total_appearances += 1
            
            # Keep only recent feature history
            if len(player.feature_history) > 20:
                player.feature_history = player.feature_history[-20:]
        
        # Create new players for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                # Check if this might be a re-appearing player
                reidentified_player_id = self.attempt_reidentification(detection)
                
                if reidentified_player_id is not None:
                    # Reactivate existing player
                    player = self.players[reidentified_player_id]
                    player.is_active = True
                    player.detections.append(detection)
                    player.feature_history.append(detection.features)
                    player.last_seen_frame = self.frame_count
                    player.total_appearances += 1
                    logger.info(f"Re-identified player {reidentified_player_id} at frame {self.frame_count}")
                else:
                    # Create new player
                    new_player = Player(
                        player_id=self.next_player_id,
                        detections=[detection],
                        feature_history=[detection.features],
                        last_seen_frame=self.frame_count,
                        total_appearances=1
                    )
                    self.players[self.next_player_id] = new_player
                    logger.info(f"Created new player {self.next_player_id} at frame {self.frame_count}")
                    self.next_player_id += 1
        
        # Deactivate players who haven't been seen recently
        for player in self.players.values():
            if (self.frame_count - player.last_seen_frame) > self.max_disappeared_frames:
                player.is_active = False
    
    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """Process a single frame and return player IDs with bounding boxes"""
        detections = self.detect_players(frame)
        self.update_players(detections)
        self.frame_count += 1
        
        # Return active player positions
        results = []
        for player in self.players.values():
            if player.is_active and player.detections:
                latest_detection = player.detections[-1]
                if latest_detection.frame_id == self.frame_count - 1:
                    results.append((player.player_id, latest_detection.bbox))
        
        return results
    
    def save_tracking_data(self, filepath: str):
        """Save tracking data for analysis"""
        tracking_data = {
            'players': {pid: {
                'total_appearances': player.total_appearances,
                'first_seen': player.detections[0].frame_id if player.detections else -1,
                'last_seen': player.last_seen_frame,
                'detection_count': len(player.detections)
            } for pid, player in self.players.items()},
            'total_frames': self.frame_count
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(tracking_data, f)
            logger.info(f"Tracking data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save tracking data: {e}")

def visualize_tracking(frame: np.ndarray, player_results: List[Tuple[int, Tuple[int, int, int, int]]]) -> np.ndarray:
    """Visualize tracking results on frame"""
    vis_frame = frame.copy()
    
    # Define colors for different players
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0)
    ]
    
    for player_id, bbox in player_results:
        x1, y1, x2, y2 = bbox
        color = colors[player_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw player ID
        label = f"Player {player_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_frame

def main():
    """Main processing pipeline"""
    # Check command line arguments
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        # Look for common video file names
        possible_files = [
            "15sec_input_720p.mp4",
            "input.mp4", 
            "test_video.mp4",
            "video.mp4"
        ]
        
        input_video = None
        for file in possible_files:
            if os.path.exists(file):
                input_video = file
                break
        
        if not input_video:
            print("Usage: python standalone_tracker.py <video_file>")
            print("Or place a video file named '15sec_input_720p.mp4' in the current directory")
            return
    
    if not os.path.exists(input_video):
        logger.error(f"Input video {input_video} not found!")
        return
    
    # Initialize tracker
    try:
        tracker = PlayerTracker(
            model_path="yolo11n.pt",
            max_disappeared_frames=30,
            similarity_threshold=0.3
        )
    except Exception as e:
        logger.error(f"Failed to initialize tracker: {e}")
        return
    
    output_video = "output_tracked.mp4"
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_video}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    logger.info(f"Processing video: {input_video}")
    logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            player_results = tracker.process_frame(frame)
            
            # Visualize results
            vis_frame = visualize_tracking(frame, player_results)
            
            # Write frame
            out.write(vis_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                active_players = len([p for p in tracker.players.values() if p.is_active])
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), Active players: {active_players}")
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        # Cleanup
        cap.release()
        out.release()
    
    # Save tracking data
    tracker.save_tracking_data("tracking_results.pkl")
    
    # Print summary
    logger.info(f"\nTracking Summary:")
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total unique players detected: {len(tracker.players)}")
    
    for player_id, player in tracker.players.items():
        first_frame = player.detections[0].frame_id if player.detections else 'N/A'
        logger.info(f"Player {player_id}: {player.total_appearances} detections, "
                   f"frames {first_frame}-{player.last_seen_frame}")
    
    logger.info(f"Output saved to: {output_video}")
    print(f"\n✓ Processing complete! Output saved to: {output_video}")

if __name__ == "__main__":
    main()
