# Feature Extraction Pipeline for Visual Perception

A modular computer vision pipeline implementing classical feature detection, description, and matching techniques for robotics and autonomous perception tasks.

This project implements a full feature extraction system inspired by SIFT/ORB pipelines and evaluates robustness under real-world distortions such as lighting changes, motion blur, and noise.

---

## 🚀 Features

- **Multi-scale feature detection**
  - Harris Corner Detection
  - Difference of Gaussians (DoG)
  - FAST keypoints

- **Scale-space representation**
  - Gaussian pyramids

- **Rotation invariance**
  - Orientation assignment for keypoints

- **Custom feature descriptors**
  - SIFT-like descriptors
  - ORB-like binary descriptors
  - BRIEF descriptors

- **Feature matching**
  - Brute-force matcher
  - FLANN-based matcher
  - Lowe’s ratio test filtering

- **Geometric verification**
  - RANSAC-based homography estimation

- **Robustness evaluation**
  - Lighting variation
  - Motion blur
  - Gaussian noise

- **Evaluation metrics**
  - Matching accuracy
  - Precision / Recall / F1-score
  - Repeatability
  - Localization error

---

## 🧠 Motivation

Feature extraction is a fundamental building block for:

- Visual SLAM
- Structure-from-Motion (SfM)
- Autonomous driving perception
- Robotics navigation systems

This project demonstrates a full end-to-end classical computer vision pipeline built from scratch to deeply understand how modern feature detectors and matchers work internally.

---

## 🏗️ Architecture Overview

The pipeline follows a standard feature extraction and matching flow:


Input Images
↓
Preprocessing (Grayscale / Normalization)
↓
Scale-Space Construction (Gaussian Pyramid)
↓
Keypoint Detection
(Harris / DoG / FAST)
↓
Keypoint Refinement & Localization
↓
Orientation Assignment
↓
Feature Descriptor Extraction
(SIFT-like / ORB / BRIEF)
↓
Feature Matching
(BF Matcher / FLANN + Ratio Test)
↓
Geometric Verification
(RANSAC Homography)
↓
Final Matches + Evaluation Metrics


---

## 📂 Project Structure


feature-extraction-pipeline/
│
├── src/ # Core implementation
│ ├── feature_detector.py
│ ├── feature_matcher.py
│ ├── scale_space.py
│ ├── descriptor.py
│ ├── orientation.py
│ └── utils.py
│
├── evaluation/ # Benchmarking & metrics
│ ├── benchmark.py
│ └── metrics.py
│
├── examples/ # Demo scripts
│ └── demo.py
│
├── tests/ # Unit tests
│ └── test_detector.py
│
├── config/ # Configuration files
│ └── config.yaml
│
├── results/ # Output visualizations
├── requirements.txt
├── setup.py
└── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/feature-extraction-pipeline.git
cd feature-extraction-pipeline

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

## ▶️ Quick Start
python examples/demo.py

## 🧪 Benchmark
python -m evaluation.benchmark --config config/config.yaml

Evaluates performance under:

Lighting changes
Motion blur
Noise injection

## 📊 Evaluation Metrics
Matching accuracy
Precision / Recall / F1-score
Feature repeatability
Localization error (pixel distance)
RANSAC inlier ratio

#🛠️ Tech Stack
Python 3.8+
OpenCV
NumPy
SciPy
Matplotlib
PyYAML
PyTest

## 🧩 Architecture Diagram (Visual)

<img width="636" height="233" alt="image" src="https://github.com/user-attachments/assets/06d43790-a8b2-4408-8f17-a956665ab48e" />


## 👤 Author

Malavika Priyesh

GitHub: https://github.com/MalavikaPriyesh
LinkedIn: https://linkedin.com/in/malavikapriyesh
## 📄 License

This project is licensed under the MIT License.
