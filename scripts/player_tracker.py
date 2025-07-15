"""
Player Re-identification System
Author: [Your Name]
Date: January 2025

This module implements a computer vision system for tracking players across video frames,
maintaining consistent IDs even when players temporarily leave the field of view.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging - learned this is important for debugging tracking issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlayerDetection:
    """Simple class to hold detection data - easier than using tuples everywhere"""
    def __init__(self, bbox, confidence, frame_num, features, center_point):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.frame_num = frame_num
        self.features = features
        self.center = center_point


class TrackedPlayer:
    """Represents a player being tracked across multiple frames"""
    def __init__(self, player_id, initial_detection):
        self.id = player_id
        self.detections = [initial_detection]
        self.feature_history = [initial_detection.features]
        self.last_seen = initial_detection.frame_num
        self.total_detections = 1
        self.active = True
        
    def update(self, detection):
        """Add new detection to this player's track"""
        self.detections.append(detection)
        self.feature_history.append(detection.features)
        self.last_seen = detection.frame_num
        self.total_detections += 1
        
        # Keep memory usage reasonable - only store last 15 feature vectors
        if len(self.feature_history) > 15:
            self.feature_history = self.feature_history[-15:]


class FeatureExtractor:
    """
    Extracts features from player bounding boxes for matching.
    Uses color histograms + spatial info - found this works better than just appearance.
    """
    
    def __init__(self):
        # Tried different bin sizes, 24 seems to work well for sports videos
        self.hist_bins = 24
        
    def get_color_features(self, image_patch):
        """Extract color histogram from player patch"""
        if image_patch.size == 0:
            return np.zeros(self.hist_bins * 3)
            
        try:
            # Convert to HSV - more robust to lighting changes than RGB
            hsv = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [self.hist_bins], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [self.hist_bins], [0, 256])
            
            # Normalize to prevent issues with different patch sizes
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
            v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)
            
            return np.concatenate([h_hist, s_hist, v_hist])
            
        except Exception as e:
            # Sometimes get weird errors with very small patches
            logger.warning(f"Color feature extraction failed: {e}")
            return np.zeros(self.hist_bins * 3)
    
    def get_position_features(self, bbox, frame_shape):
        """Get normalized position and size features"""
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Normalize everything to [0,1] range
        center_x = (x1 + x2) / (2.0 * frame_w)
        center_y = (y1 + y2) / (2.0 * frame_h)
        width = (x2 - x1) / frame_w
        height = (y2 - y1) / frame_h
        aspect = width / (height + 1e-6)  # avoid division by zero
        
        return np.array([center_x, center_y, width, height, aspect])
    
    def extract_features(self, frame, bbox):
        """Main feature extraction - combines color and position info"""
        x1, y1, x2, y2 = bbox
        
        # Make sure bbox is valid
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Extract the player patch
        patch = frame[y1:y2, x1:x2]
        
        # Resize to standard size for consistent features
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            patch = cv2.resize(patch, (48, 96))  # roughly 1:2 aspect ratio for people
        else:
            patch = np.zeros((96, 48, 3), dtype=np.uint8)
        
        # Get both types of features
        color_feats = self.get_color_features(patch)
        pos_feats = self.get_position_features(bbox, frame.shape)
        
        # Combine them
        return np.concatenate([color_feats, pos_feats])


class PlayerTracker:
    """
    Main tracking class. Handles detection, matching, and re-identification.
    
    The basic approach:
    1. Detect people in each frame using YOLO
    2. Extract features from each detection
    3. Match detections to existing tracks using Hungarian algorithm
    4. Handle new players and re-identification of lost players
    """
    
    def __init__(self, model_path="yolo11n.pt"):
        # Load YOLO model
        try:
            self.detector = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        self.feature_extractor = FeatureExtractor()
        
        # Tracking state
        self.players = {}  # dict of player_id -> TrackedPlayer
        self.next_id = 1
        self.current_frame = 0
        
        # Tuning parameters - adjusted these based on testing
        self.max_missing_frames = 25  # how long to keep inactive players
        self.match_threshold = 0.35   # similarity threshold for matching
        self.reidentify_threshold = 0.45  # higher threshold for re-identification
        
    def detect_people(self, frame):
        """Run YOLO detection and return list of PlayerDetection objects"""
        detections = []
        
        try:
            # Run detection - only look for people (class 0)
            results = self.detector(frame, classes=[0], verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get bbox coordinates and confidence
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        x1, y1, x2, y2 = coords
                        
                        # Filter out low confidence and tiny detections
                        if conf > 0.4 and (x2-x1) > 15 and (y2-y1) > 30:
                            # Extract features for this detection
                            features = self.feature_extractor.extract_features(frame, coords)
                            center = ((x1+x2)/2, (y1+y2)/2)
                            
                            detection = PlayerDetection(
                                bbox=coords,
                                confidence=conf,
                                frame_num=self.current_frame,
                                features=features,
                                center_point=center
                            )
                            detections.append(detection)
                            
        except Exception as e:
            logger.warning(f"Detection failed on frame {self.current_frame}: {e}")
            
        return detections
    
    def compute_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        if len(features1) != len(features2) or len(features1) == 0:
            return 0.0
            
        try:
            # Split into color and position features
            color1, pos1 = features1[:-5], features1[-5:]
            color2, pos2 = features2[:-5], features2[-5:]
            
            # Color similarity using cosine distance
            color_sim = 1.0 - cosine(color1, color2)
            if np.isnan(color_sim):
                color_sim = 0.0
                
            # Position similarity using euclidean distance
            pos_dist = np.linalg.norm(pos1 - pos2)
            pos_sim = 1.0 / (1.0 + pos_dist)
            
            # Weighted combination - color is more important than position
            return 0.75 * color_sim + 0.25 * pos_sim
            
        except Exception:
            return 0.0
    
    def match_detections_to_players(self, detections):
        """Use Hungarian algorithm to match detections to existing players"""
        if not detections or not self.players:
            return {}
            
        # Get list of active players
        active_players = [p for p in self.players.values() if p.active]
        if not active_players:
            return {}
        
        # Build cost matrix
        num_players = len(active_players)
        num_detections = len(detections)
        cost_matrix = np.ones((num_players, num_detections))
        
        for i, player in enumerate(active_players):
            # Use average of recent features for more stable matching
            recent_features = player.feature_history[-3:]  # last 3 frames
            avg_features = np.mean(recent_features, axis=0)
            
            for j, detection in enumerate(detections):
                similarity = self.compute_similarity(avg_features, detection.features)
                cost_matrix[i, j] = 1.0 - similarity  # convert to cost
        
        # Solve assignment problem
        try:
            player_indices, detection_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            logger.warning(f"Hungarian algorithm failed: {e}")
            return {}
        
        # Filter out poor matches
        matches = {}
        for p_idx, d_idx in zip(player_indices, detection_indices):
            cost = cost_matrix[p_idx, d_idx]
            if cost < (1.0 - self.match_threshold):
                player_id = active_players[p_idx].id
                matches[player_id] = d_idx
                
        return matches
    
    def try_reidentification(self, detection):
        """Try to match detection to an inactive player (re-identification)"""
        inactive_players = [p for p in self.players.values() if not p.active]
        
        best_match = None
        best_similarity = 0.0
        
        for player in inactive_players:
            # Compare against multiple recent features
            similarities = []
            for old_features in player.feature_history[-8:]:  # check last 8 features
                sim = self.compute_similarity(old_features, detection.features)
                similarities.append(sim)
            
            if similarities:
                max_sim = max(similarities)
                if max_sim > best_similarity and max_sim > self.reidentify_threshold:
                    best_similarity = max_sim
                    best_match = player.id
        
        return best_match
    
    def update_tracks(self, detections):
        """Main tracking update - match detections and update player tracks"""
        # Match detections to existing active players
        matches = self.match_detections_to_players(detections)
        matched_detection_ids = set(matches.values())
        
        # Update matched players
        for player_id, detection_idx in matches.items():
            detection = detections[detection_idx]
            self.players[player_id].update(detection)
        
        # Handle unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_ids:
                # First try re-identification
                reidentified_id = self.try_reidentification(detection)
                
                if reidentified_id is not None:
                    # Reactivate the player
                    player = self.players[reidentified_id]
                    player.active = True
                    player.update(detection)
                    logger.info(f"Re-identified player {reidentified_id} at frame {self.current_frame}")
                else:
                    # Create new player
                    new_player = TrackedPlayer(self.next_id, detection)
                    self.players[self.next_id] = new_player
                    logger.info(f"New player {self.next_id} detected at frame {self.current_frame}")
                    self.next_id += 1
        
        # Deactivate players that haven't been seen recently
        for player in self.players.values():
            frames_missing = self.current_frame - player.last_seen
            if frames_missing > self.max_missing_frames:
                player.active = False
    
    def process_frame(self, frame):
        """Process a single frame - main entry point"""
        # Detect people in this frame
        detections = self.detect_people(frame)
        
        # Update tracking
        self.update_tracks(detections)
        
        # Increment frame counter
        self.current_frame += 1
        
        # Return current active player positions for visualization
        active_tracks = []
        for player in self.players.values():
            if player.active and player.detections:
                latest_detection = player.detections[-1]
                if latest_detection.frame_num == self.current_frame - 1:
                    active_tracks.append((player.id, latest_detection.bbox))
        
        return active_tracks
    
    def save_results(self, filename):
        """Save tracking results to file"""
        results = {
            'total_frames': self.current_frame,
            'players': {}
        }
        
        for player_id, player in self.players.items():
            results['players'][player_id] = {
                'total_detections': player.total_detections,
                'first_frame': player.detections[0].frame_num if player.detections else -1,
                'last_frame': player.last_seen,
                'detection_count': len(player.detections)
            }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def draw_tracks(frame, tracks):
    """Draw bounding boxes and IDs on frame"""
    # Different colors for different players
    colors = [
        (0, 0, 255),    # red
        (0, 255, 0),    # green  
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
        (128, 0, 128),  # purple
        (255, 165, 0),  # orange
        (0, 128, 0),    # dark green
        (128, 128, 0),  # olive
    ]
    
    output_frame = frame.copy()
    
    for player_id, bbox in tracks:
        x1, y1, x2, y2 = bbox
        color = colors[player_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw player ID label
        label = f"P{player_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Background rectangle for label
        cv2.rectangle(output_frame, 
                     (x1, y1 - label_size[1] - 8), 
                     (x1 + label_size[0] + 4, y1), 
                     color, -1)
        
        # Label text
        cv2.putText(output_frame, label, (x1 + 2, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output_frame


def main():
    """Main function to run the tracking system"""
    import sys
    
    # Get input video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Look for common video names
        candidates = ["15sec_input_720p.mp4", "input.mp4", "test.mp4", "video.mp4"]
        video_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                video_path = candidate
                break
        
        if video_path is None:
            print("Usage: python player_tracker.py <video_file>")
            print("Or place video file as '15sec_input_720p.mp4' in current directory")
            return
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return
    
    print(f"Processing video: {video_path}")
    
    # Initialize tracker
    try:
        tracker = PlayerTracker()
    except Exception as e:
        print(f"Failed to initialize tracker: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video
    output_path = "tracked_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frame by frame
    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run tracking on this frame
            tracks = tracker.process_frame(frame)
            
            # Draw tracking results
            output_frame = draw_tracks(frame, tracks)
            
            # Write to output video
            out.write(output_frame)
            
            frame_num += 1
            
            # Progress update every 30 frames
            if frame_num % 30 == 0:
                active_count = len([p for p in tracker.players.values() if p.active])
                progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
                print(f"Frame {frame_num}/{total_frames} ({progress:.1f}%) - {active_count} active players")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        cap.release()
        out.release()
    
    # Save tracking results
    tracker.save_results("tracking_data.pkl")
    
    # Print summary
    print(f"\nTracking completed!")
    print(f"Processed {frame_num} frames")
    print(f"Detected {len(tracker.players)} unique players")
    print(f"Output video: {output_path}")
    
    # Show per-player statistics
    for player_id, player in tracker.players.items():
        first_frame = player.detections[0].frame_num if player.detections else "N/A"
        print(f"Player {player_id}: {player.total_detections} detections "
              f"(frames {first_frame} to {player.last_seen})")


if __name__ == "__main__":
    main()
