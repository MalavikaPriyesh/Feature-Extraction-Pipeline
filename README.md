# Feature Extraction Pipeline for Visual Perception

A modular computer vision pipeline implementing classical feature detection, description, and matching techniques for robotics and autonomous perception tasks.

This project implements a full feature extraction system inspired by SIFT/ORB pipelines and evaluates robustness under real-world distortions such as lighting changes, motion blur, and noise.

---

## 🚀 Features

- Multi-scale feature detection:
  - Harris Corner Detection
  - Difference of Gaussians (DoG)
  - FAST keypoints

- Scale-space construction using Gaussian pyramids
- Rotation invariance via orientation assignment

- Custom feature descriptors:
  - SIFT-like descriptors
  - ORB-like binary descriptors
  - BRIEF descriptors

- Feature matching:
  - Brute-force matcher
  - FLANN-based matcher
  - Lowe’s ratio test filtering

- Geometric verification:
  - RANSAC-based homography estimation

- Evaluation under distortions:
  - Lighting variation
  - Motion blur
  - Gaussian noise

- Metrics:
  - Matching accuracy
  - Precision / Recall / F1-score
  - Repeatability
  - Localization error

---

## 🧠 Motivation

Feature extraction is a fundamental component in:

- Visual SLAM
- Structure-from-Motion (SfM)
- Autonomous driving perception
- Robotics navigation systems

---

## 📂 Project Structure


feature-extraction-pipeline/
├── src/
├── evaluation/
├── examples/
├── tests/
├── config/
├── results/
├── requirements.txt
├── setup.py
└── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/feature-extraction-pipeline.git
cd feature-extraction-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
▶️ Quick Start
python examples/demo.py
🧪 Benchmark
python -m evaluation.benchmark --config config/config.yaml
📊 Metrics
Matching accuracy
Precision / Recall / F1-score
Repeatability
Localization error
🛠️ Tech Stack
Python
OpenCV
NumPy
SciPy
Matplotlib
PyYAML
PyTest
👤 Author

Malavika Priyesh
GitHub: https://github.com/yourusername
LinkedIn: https://linkedin.com/in/your-profile

📄 License

MIT License
