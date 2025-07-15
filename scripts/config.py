"""Configuration settings for player re-identification system"""

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'best.pt',  
    'confidence_threshold': 0.5,
    'min_bbox_width': 20,
    'min_bbox_height': 40,
    'target_classes': [0],  # Person class
}

# Tracking Configuration
TRACKING_CONFIG = {
    'max_disappeared_frames': 30,  # Frames before considering player as disappeared
    'similarity_threshold': 0.3,   # Minimum similarity for matching
    'reidentification_threshold': 0.4,  # Higher threshold for re-identification
    'feature_history_length': 20,  # Number of recent features to keep
    'matching_features_count': 5,  # Number of recent features for matching
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'color_bins': 32,           # Number of bins for color histogram
    'patch_size': (64, 128),    # Standard size for extracted patches
    'color_weight': 0.7,        # Weight for color features in similarity
    'spatial_weight': 0.3,      # Weight for spatial features in similarity
}

# Video Processing Configuration
VIDEO_CONFIG = {
    'input_video': '15sec_input_720p.mp4',
    'output_video': 'output_tracked.mp4',
    'codec': 'mp4v',
    'log_interval': 30,  # Log progress every N frames
}

# Visualization Configuration
VIS_CONFIG = {
    'colors': [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0)
    ],
    'bbox_thickness': 2,
    'font_scale': 0.6,
    'font_thickness': 2,
}
