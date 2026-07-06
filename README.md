# Squat Tracker & Visual Validator

A Python-based computer vision application that performs real-time validation of squat form, monitors posture, and counts repetitions using **MediaPipe Pose** and **OpenCV**.

---

## Features

- **Dual-Criteria Depth Detection:** A repetition is validated for proper depth when:
  - The glutes/hip crease drops below the top of the knee (`hip[1] > knee[1]`), **OR**
  - The knee angle (hip-knee-ankle) drops below $90^\circ$.
- **Real-Time HUD Overlay:** A clean, semi-transparent heads-up display overlaying:
  - Total valid repetition count.
  - Live depth status (`DEPTH: GOOD` vs `DEPTH: SHALLOW`).
  - Torso lean analysis (`POSTURE: OK` vs `WARN: KEEP CHEST UP!`).
- **Posture Analysis:** Tracks the torso lean angle relative to the vertical axis. Generates a warning if the torso tilts forward past $40^\circ$.
- **Dynamic Visual Cues:** Joint indicators and skeletal connection lines dynamically switch to **Neon Green** when proper depth is achieved, and remain **Red/Blue** otherwise.
- **Mac-Compatible Exporter:** Automatically exports the processed analysis video next to the input video as `<filename>_squat_analysis.mov` using the QuickTime-friendly H.264 (`avc1`) codec.

---

## Installation & Setup

1. **Prerequisites:** Ensure you have Python 3.11 installed.
2. **Set up Virtual Environment:**
   ```bash
   python3.11 -m venv .venv
   ```
3. **Install Dependencies:**
   ```bash
   .venv/bin/pip install -r requirements.txt
   ```

---

## Usage

Run the analysis script by pointing it to any local video file (defaults to `squat120.mp4` if no argument is passed):

```bash
.venv/bin/python main.py [path/to/video.mp4]
```

### Controls
* Press **`q`** in the window to quit the analysis early.

---

## Project Structure

* `main.py` - Core application logic, MediaPipe processing loop, and video exporter.
* `requirements.txt` - Python package dependencies.
* `.gitignore` - Workspace ignore rules (excludes virtual environment and generated video files).
