# ThAIrpy – AI and AR-Enhanced Physical Therapy Project

https://img.shields.io/badge/Author-Malek_Kchaou-yellow


ThAIrpy is a research-driven project aimed at enhancing physical therapy using Artificial Intelligence (AI) and Augmented Reality (AR). The goal is to develop an immersive, data-driven system that assists both patients and therapists through real-time feedback, guided exercises, and smart progress tracking.

---

## Project Overview

ThAIrpy tackles key issues in physical therapy, including:
- Limited access in rural or underserved areas
- High cost of traditional therapy sessions
- Low patient adherence due to lack of engagement
- Time-consuming documentation for therapists
- Difficulty tracking patient progress

---

## Key Features

### For Patients
- **AR-Guided Exercises** – Visual cues help ensure proper form.
- **Real-Time AI Corrections** – Detects and corrects movement errors.
- **Gamified Rehab** – Adds challenges and achievements to increase engagement.
- **Progress Tracking** – Visual dashboards to monitor improvement.
- **Remote Access** – Perform therapy from home using AR or video guidance.

### For Therapists
- **AI-Assisted Plan Creation** – Generate and adapt therapy plans based on performance data.
- **Automated Documentation** – Includes voice-to-text and smart templates.
- **Productivity Dashboards** – Track patient outcomes and therapist efficiency.
- **Patient Retention Tools** – Alerts, reminders, and visual feedback to reduce dropouts.

---

## Development Roadmap

1. **Research & Ideation**
   - Study Mediapipe, OpenPose, ARCore, and physical therapy workflows.

2. **Prototype**
   - Build a system for motion tracking and feedback.

3. **Feature Expansion**
   - Add AR overlays, wearable integration, dashboards, and patient data access.

4. **Testing**
   - Validate features with therapists and patients; collect feedback.

---

## Technologies

- TensorFlow / TensorFlow Lite
- TensorFlow Hub (MoveNet models)
- OpenCV & Matplotlib
- Python (NumPy, ImageIO, PIL)
- Optional: ARCore / ARKit integration (planned)

---
## Current Progress
The project has successfully implemented an early-stage motion tracking system using TensorFlow MoveNet for human pose estimation. All the used scripts are in the **'MoveNet'** folder. Here’s what’s been built so far:


### `movenet.py`: Pose Estimation & Visualization
The `movenet.py` script implements the core pose detection pipeline using TensorFlow's MoveNet model:

**Pose Detection Pipeline**
- Integrated MoveNet Lightning/Thunder models (TensorFlow Hub and TFLite formats).

- Supported both real-time and static image analysis.

- Applied dynamic cropping to focus on relevant body areas.

- Converted model output into pixel-accurate keypoints.

**Visualization**
- Developed a custom drawing function to overlay skeleton keypoints and edges on input images using matplotlib.

- Added support for animated GIF output for multiple frames (with imageio).

**Technical Capabilities**
- Confidence filtering of keypoints (threshold-based).

- Skeleton edge rendering with distinct colors.

- Adaptive crop region computation based on torso visibility and pose.

- Support for JPEG image processing with fallback for alpha channels (RGBA → RGB).

- Initial support for progress tracking and pose correction (framework in place).

**Testing**
- Verified pose estimation accuracy using sample static images.

- Output tested using matplotlib displays and preview animations.

### `preprocessor.py`: Dataset Preparation Pipeline

The `preprocessor.py` script enables transformation of raw pose image datasets into a format suitable for training machine learning models:

**Dataset Splitting**
- Automatically divides images into **training** and **testing** folders
  
- Supports a customizable split ratio (default is 60% training)
  
- Maintains pose category structure for supervised classification tasks

**Keypoint Extraction**
- Extracts 17 keypoints per image (x, y coordinates + confidence scores)
  
- Uses MoveNet to infer keypoint data per image

**CSV Generation**
- Generates `train.csv` and `test.csv` files
  
- Each row contains:
   - Pose category label
   - Image filename
   - Flattened keypoint values (x1, y1, c1, ..., x17, y17, c17)

This pipeline creates a clean and organized dataset ready for pose classification, exercise recognition, or movement analysis.

---

## Tech Stack

- TensorFlow / TensorFlow Lite
- TensorFlow Hub (MoveNet models)
- OpenCV & Matplotlib
- Python (NumPy, ImageIO, PIL)
- Optional: ARCore / ARKit integration (planned)

---

## Demo Resource

Example motion tracking prototype:  
[Arduino + Python OpenCV Tracker](https://github.com/kensunjaya/arduino-motion-tracking-with-py-cv2)

---

## Status

**Currently in prototyping and preprocessing phase.**

---

## Contact

For questions or collaboration inquiries, feel free to open an issue or contact the maintainer.
Let me know if you want me to generate a version that includes an image, badge, or sample output!











  
---

## Status

**Currently in prototype and research phase.**

---

## Contact

For questions or collaborations, feel free to reach out via GitHub issues.


