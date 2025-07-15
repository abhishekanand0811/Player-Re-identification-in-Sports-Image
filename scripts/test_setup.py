"""Test script to verify setup and dependencies"""

import sys
import os
import logging

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found. Install with: pip install opencv-python")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO available")
    except ImportError:
        print("✗ Ultralytics not found. Install with: pip install ultralytics")
        return False
    
    try:
        from scipy.optimize import linear_sum_assignment
        print("✓ SciPy available")
    except ImportError:
        print("✗ SciPy not found. Install with: pip install scipy")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib available")
    except ImportError:
        print("✗ Matplotlib not found. Install with: pip install matplotlib")
        return False
    
    return True

def test_video_file():
    """Test if video file exists and is readable"""
    print("\nTesting video file...")
    
    video_files = [
        "15sec_input_720p.mp4",
        "input.mp4",
        "test_video.mp4"
    ]
    
    for video_file in video_files:
        if os.path.exists(video_file):
            print(f"✓ Found video file: {video_file}")
            
            # Test if video can be opened
            try:
                import cv2
                cap = cv2.VideoCapture(video_file)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    print(f"  - Resolution: {width}x{height}")
                    print(f"  - FPS: {fps}")
                    print(f"  - Total frames: {frame_count}")
                    print(f"  - Duration: {frame_count/fps:.2f} seconds")
                    
                    cap.release()
                    return video_file
                else:
                    print(f"✗ Cannot open video file: {video_file}")
            except Exception as e:
                print(f"✗ Error testing video file: {e}")
    
    print("✗ No valid video file found")
    print("Please place your video file in the current directory and name it:")
    print("  - 15sec_input_720p.mp4 (preferred)")
    print("  - input.mp4")
    print("  - test_video.mp4")
    return None

def test_yolo_model():
    """Test YOLO model loading"""
    print("\nTesting YOLO model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolo11n.pt')  
        print("✓ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading YOLO model: {e}")
        return False

def main():
    """Run all tests"""
    print("Player Re-Identification Setup Test")
    print("=" * 40)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test video file
    video_file = test_video_file()
    
    # Test YOLO model
    model_ok = test_yolo_model()
    
    print("\n" + "=" * 40)
    print("SETUP TEST SUMMARY")
    print("=" * 40)
    
    if deps_ok and video_file and model_ok:
        print("✓ All tests passed! You're ready to run the tracking system.")
        print(f"\nTo run tracking on {video_file}, update the input_video variable in player_reidentification.py")
        print("Then run: python run_tracking.py")
    else:
        print("✗ Some tests failed. Please fix the issues above before running the tracking system.")
        
        if not deps_ok:
            print("\n1. Install missing dependencies:")
            print("   pip install -r requirements.txt")
        
        if not video_file:
            print("\n2. Add a video file to the current directory")
        
        if not model_ok:
            print("\n3. Check your internet connection for YOLO model download")

if __name__ == "__main__":
    main()
