"""Main script to run the player re-identification system"""

import sys
import os
import logging
from player_reidentification import main as run_tracking
from evaluation_metrics import evaluate_tracking_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracking.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    logger.info("Starting Player Re-Identification System")
    
    tracking_successful = False
    
    try:
        # Run the tracking
        logger.info("Phase 1: Running player tracking...")
        run_tracking()
        tracking_successful = True
        
    except KeyboardInterrupt:
        logger.info("Tracking interrupted by user")
        tracking_successful = True  
        
    except Exception as e:
        logger.error(f"Error during tracking: {str(e)}")
        logger.error("Tracking phase failed. Please check your video file path and dependencies.")
        return
    
    if tracking_successful and os.path.exists("tracking_results.pkl"):
        try:
            logger.info("Phase 2: Evaluating tracking results...")
            evaluate_tracking_results()
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            logger.warning("Tracking completed but evaluation failed.")
    else:
        logger.warning("Skipping evaluation - tracking results not found.")
    
    if tracking_successful:
        logger.info("Player re-identification completed!")
        
        print("\n" + "="*50)
        print("PLAYER RE-IDENTIFICATION COMPLETE")
        print("="*50)
        print("Generated files:")
        if os.path.exists("output_tracked.mp4"):
            print("- output_tracked.mp4: Video with player tracking")
        if os.path.exists("tracking_results.pkl"):
            print("- tracking_results.pkl: Raw tracking data")
        if os.path.exists("tracking_report.txt"):
            print("- tracking_report.txt: Performance analysis")
        if os.path.exists("tracking_timeline.png"):
            print("- tracking_timeline.png: Timeline visualization")
        if os.path.exists("detection_distribution.png"):
            print("- detection_distribution.png: Detection statistics")
        print("- tracking.log: Detailed execution log")
    else:
        logger.error("Player re-identification failed. Check the log for details.")

if __name__ == "__main__":
    main()
