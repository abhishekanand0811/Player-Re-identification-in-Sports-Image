"""
Analysis script for player tracking results
Generates visualizations and statistics from tracking data
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def load_tracking_data(filename="tracking_data.pkl"):
    """Load tracking results from pickle file"""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run tracking first.")
        return None
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_tracking_performance(data):
    """Analyze and print tracking statistics"""
    if data is None:
        return
    
    players = data['players']
    total_frames = data['total_frames']
    
    print("=== TRACKING ANALYSIS ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Unique players detected: {len(players)}")
    
    # Calculate statistics
    detection_counts = [p['detection_count'] for p in players.values()]
    frame_spans = [p['last_frame'] - p['first_frame'] + 1 for p in players.values() if p['first_frame'] >= 0]
    
    print(f"\nDetection Statistics:")
    print(f"  Average detections per player: {np.mean(detection_counts):.1f}")
    print(f"  Min detections: {min(detection_counts)}")
    print(f"  Max detections: {max(detection_counts)}")
    
    if frame_spans:
        print(f"\nTracking Duration:")
        print(f"  Average tracking span: {np.mean(frame_spans):.1f} frames")
        print(f"  Longest track: {max(frame_spans)} frames")
        print(f"  Shortest track: {min(frame_spans)} frames")
    
    print(f"\nPer-Player Breakdown:")
    for pid, player in players.items():
        first = player['first_frame']
        last = player['last_frame']
        count = player['detection_count']
        span = last - first + 1 if first >= 0 else 0
        coverage = (count / span * 100) if span > 0 else 0
        
        print(f"  Player {pid}: {count} detections over {span} frames ({coverage:.1f}% coverage)")

def create_timeline_plot(data, save_path="timeline.png"):
    """Create timeline visualization showing when each player was active"""
    if data is None:
        return
    
    players = data['players']
    total_frames = data['total_frames']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create timeline bars for each player
    y_pos = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(players)))
    
    for i, (pid, player) in enumerate(players.items()):
        if player['first_frame'] >= 0:
            start = player['first_frame']
            end = player['last_frame']
            duration = end - start + 1
            
            ax.barh(y_pos, duration, left=start, height=0.6, 
                   color=colors[i], alpha=0.7, label=f'Player {pid}')
            y_pos += 1
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Player')
    ax.set_title('Player Tracking Timeline')
    ax.set_xlim(0, total_frames)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis labels
    ax.set_yticks(range(len(players)))
    ax.set_yticklabels([f'P{pid}' for pid in players.keys()])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Timeline plot saved to {save_path}")

def create_detection_histogram(data, save_path="detections.png"):
    """Create histogram of detection counts per player"""
    if data is None:
        return
    
    players = data['players']
    detection_counts = [p['detection_count'] for p in players.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(detection_counts, bins=max(10, len(players)//2), 
           color='skyblue', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Number of Detections')
    ax.set_ylabel('Number of Players')
    ax.set_title('Distribution of Detection Counts')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_detections = np.mean(detection_counts)
    ax.axvline(mean_detections, color='red', linestyle='--', 
              label=f'Mean: {mean_detections:.1f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Detection histogram saved to {save_path}")

def main():
    """Main analysis function"""
    print("Loading tracking results...")
    data = load_tracking_data()
    
    if data is None:
        return
    
    # Print analysis
    analyze_tracking_performance(data)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_timeline_plot(data)
    create_detection_histogram(data)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
