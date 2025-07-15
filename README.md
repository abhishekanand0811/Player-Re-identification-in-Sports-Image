# 🏃‍♂️ Player Re-identification in Sports Image

**AI-powered computer vision system for tracking players in sports footage with robust re-identification capabilities**

## 📖 Overview

This project implements a visual tracking system that detects and re-identifies players across video frames using YOLOv11n and handcrafted features. It includes detection, tracking, re-identification, analytics, and visualizations.

## 🧠 Features
✅ Player detection using YOLOv8

✅ Appearance + spatial feature-based re-identification

✅ Handles occlusion and re-entry using cosine similarity

✅ Supports multiple input videos

✅ Generates tracking reports and statistics

✅ Visualization tools: bounding boxes, timelines, histograms

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### ⚙️ Choosing Between run_tracking.py, run_tracker.py, and player_reidentification.py
Each script serves a slightly different purpose and provides flexibility depending on your use case.

# 1. Run Files

✅ run_tracking.py (Recommended for most users)

* Purpose: Runs the full player tracking pipeline using a user-friendly video selection setup.

* How it works:

    * Automatically lists videos from the input_videos/ folder.

    * Prompts you to select a video to process.

    * Saves the annotated output video in the output_videos/ folder.

* Best for: General usage and testing with different input videos without modifying code.
---

Run it with:
```bash
python run_tracking.py
```
---

### 🧪 player_reidentification.py (Main logic module)

* Purpose: Contains the core tracking and re-identification logic.

* How it works:

    * Hardcoded to use a single video: 15sec_input_720p.mp4.

    * Outputs the result as output_tracked.mp4.

* Best for: Developers/debuggers who want to test or modify the tracking logic directly.
---

Run it with:
```bash
python player_reidentification.py
```
---

### 🧪 run_tracker.py (Legacy/Alternate runner)

* Purpose: Previously used to select and run videos manually (like run_tracking.py) but less maintained.

* How it works:

    * Similar selection menu like run_tracking.py.

    * Calls an older version of the player tracking setup (player_tracker.py, now deprecated).

* Best for: Reference only. Not recommended unless you’re comparing with older versions.
---

Run it with:
``` bash
python run_tracker.py
```
---

# 2. Analyze Tracking Results

``` bash
python analyze_results.py
```
* Prints tracking stats to the console

* Generates:

    * timeline.png: Player appearance over time

    * detections.png: Detection frequency histogram
---
# 3. Evaluate Performance
``` bash
python evaluation_metrics.py
```
* Prints a detailed tracking report

* Saves:

    * tracking_report.txt

    * tracking_timeline.png

    * detection_distribution.png

---

## 📝 License 
MIT License

---

*⭐ If you found this project helpful, please give it a star! ⭐*

*Built with ❤️ by Abhishek Anand*

