import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class TrackingEvaluator:
    """Evaluate tracking performance and generate analytics"""
    
    def __init__(self, tracking_data_path: str):
        with open(tracking_data_path, 'rb') as f:
            self.tracking_data = pickle.load(f)
    
    def calculate_tracking_metrics(self) -> Dict:
        """Calculate various tracking performance metrics"""
        players = self.tracking_data['players']
        total_frames = self.tracking_data['total_frames']
        
        metrics = {
            'total_unique_players': len(players),
            'total_frames': total_frames,
            'avg_detections_per_player': np.mean([p['detection_count'] for p in players.values()]),
            'player_longevity': {},
            'frame_coverage': {}
        }
        
        # Calculate player longevity (how long each player was tracked)
        for pid, player in players.items():
            longevity = player['last_seen'] - player['first_seen'] + 1
            metrics['player_longevity'][pid] = longevity
        
        # Calculate frame coverage (percentage of frames each player appears in)
        for pid, player in players.items():
            coverage = (player['detection_count'] / total_frames) * 100
            metrics['frame_coverage'][pid] = coverage
        
        return metrics
    
    def generate_tracking_report(self) -> str:
        """Generate a comprehensive tracking report"""
        metrics = self.calculate_tracking_metrics()
        
        report = f"""
PLAYER RE-IDENTIFICATION TRACKING REPORT
========================================

Overall Statistics:
- Total unique players detected: {metrics['total_unique_players']}
- Total frames processed: {metrics['total_frames']}
- Average detections per player: {metrics['avg_detections_per_player']:.2f}

Player-wise Analysis:
"""
        
        players = self.tracking_data['players']
        for pid in sorted(players.keys()):
            player = players[pid]
            longevity = metrics['player_longevity'][pid]
            coverage = metrics['frame_coverage'][pid]
            
            report += f"""
Player {pid}:
  - First appearance: Frame {player['first_seen']}
  - Last appearance: Frame {player['last_seen']}
  - Total detections: {player['detection_count']}
  - Tracking longevity: {longevity} frames
  - Frame coverage: {coverage:.1f}%
"""
        
        return report
    
    def plot_tracking_timeline(self, save_path: str = "tracking_timeline.png"):
        """Create a timeline visualization of player appearances"""
        players = self.tracking_data['players']
        total_frames = self.tracking_data['total_frames']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(players)))
        
        for i, (pid, player) in enumerate(sorted(players.items())):
            # Create timeline bars
            start_frame = player['first_seen']
            end_frame = player['last_seen']
            
            ax.barh(i, end_frame - start_frame + 1, left=start_frame, 
                   height=0.6, color=colors[i], alpha=0.7, 
                   label=f'Player {pid}')
        
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Player ID')
        ax.set_title('Player Tracking Timeline')
        ax.set_xlim(0, total_frames)
        ax.set_yticks(range(len(players)))
        ax.set_yticklabels([f'Player {pid}' for pid in sorted(players.keys())])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Timeline plot saved to: {save_path}")
    
    def plot_detection_distribution(self, save_path: str = "detection_distribution.png"):
        """Plot distribution of detections per player"""
        players = self.tracking_data['players']
        
        detection_counts = [p['detection_count'] for p in players.values()]
        player_ids = list(players.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        ax1.bar(range(len(player_ids)), detection_counts, 
               color='skyblue', alpha=0.7)
        ax1.set_xlabel('Player ID')
        ax1.set_ylabel('Number of Detections')
        ax1.set_title('Detections per Player')
        ax1.set_xticks(range(len(player_ids)))
        ax1.set_xticklabels([f'P{pid}' for pid in player_ids])
        
        # Histogram
        ax2.hist(detection_counts, bins=10, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Number of Detections')
        ax2.set_ylabel('Number of Players')
        ax2.set_title('Distribution of Detection Counts')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detection distribution plot saved to: {save_path}")

def evaluate_tracking_results():
    """Main evaluation function"""
    evaluator = TrackingEvaluator("tracking_results.pkl")
    
    # Generate report
    report = evaluator.generate_tracking_report()
    print(report)
    
    # Save report to file
    with open("tracking_report.txt", "w") as f:
        f.write(report)
    
    # Generate visualizations
    evaluator.plot_tracking_timeline()
    evaluator.plot_detection_distribution()
    
    logger.info("Evaluation complete! Check tracking_report.txt and generated plots.")

if __name__ == "__main__":
    evaluate_tracking_results()
