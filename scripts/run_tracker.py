"""
Simple runner script for the player tracking system
Usage: python run_tracker.py [video_file]
"""

import os
import sys
from player_tracker import main as run_tracking
from analyze_results import main as run_analysis

def check_dependencies():
    """Check if required packages are installed"""
    required = ['cv2', 'numpy', 'ultralytics', 'scipy', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True

def find_video_file():
    """Look for video files in current directory"""
    extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in extensions):
            video_files.append(file)
    
    return video_files

def main():
    print("Player Re-identification System")
    print("=" * 35)
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Check for video file
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        if not os.path.exists(video_file):
            print(f"Error: Video file '{video_file}' not found")
            return
    else:
        # Look for video files
        video_files = find_video_file()
        if not video_files:
            print("No video files found in current directory")
            print("Usage: python run_tracker.py <video_file>")
            return
        elif len(video_files) == 1:
            video_file = video_files[0]
            print(f"Found video file: {video_file}")
        else:
            print("Multiple video files found:")
            for i, vf in enumerate(video_files):
                print(f"  {i+1}. {vf}")
            
            try:
                choice = int(input("Select video file (number): ")) - 1
                video_file = video_files[choice]
            except (ValueError, IndexError):
                print("Invalid selection")
                return
    
    print(f"\nProcessing: {video_file}")
    
    # Update sys.argv so the tracking script can find the video
    sys.argv = ['player_tracker.py', video_file]
    
    try:
        # Run tracking
        print("\nStep 1: Running player tracking...")
        run_tracking()
        
        # Run analysis if tracking data exists
        if os.path.exists("tracking_data.pkl"):
            print("\nStep 2: Analyzing results...")
            run_analysis()
        
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE!")
        print("=" * 50)
        print("Generated files:")
        if os.path.exists("tracked_output.mp4"):
            print("  - tracked_output.mp4 (video with tracking overlay)")
        if os.path.exists("tracking_data.pkl"):
            print("  - tracking_data.pkl (raw tracking data)")
        if os.path.exists("timeline.png"):
            print("  - timeline.png (player timeline visualization)")
        if os.path.exists("detections.png"):
            print("  - detections.png (detection statistics)")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        print("Check the log messages above for more details")

if __name__ == "__main__":
    main()
