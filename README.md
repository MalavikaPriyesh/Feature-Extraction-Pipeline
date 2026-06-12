# Feature Extraction Pipeline for Visual Perception

> A modular computer vision pipeline implementing classical feature detection, description, and matching techniques for robotics and autonomous perception tasks.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

This project implements a full feature extraction system inspired by SIFT/ORB pipelines and evaluates robustness under real-world distortions such as lighting changes, motion blur, and noise.

---

## Features

### Multi-Scale Feature Detection
- Harris Corner Detection
- Difference of Gaussians (DoG)
- FAST keypoints
- Scale-space construction using Gaussian pyramids

### Invariance
- Rotation invariance via orientation assignment

### Feature Descriptors
- SIFT-like descriptors
- ORB-like binary descriptors
- BRIEF descriptors

### Feature Matching
- Brute-force matcher
- FLANN-based matcher
- Lowe's ratio test filtering

### Geometric Verification
- RANSAC-based homography estimation

### Robustness Evaluation
- Lighting variation
- Motion blur
- Gaussian noise

### Evaluation Metrics
- Matching accuracy
- Precision / Recall / F1-score
- Repeatability
- Localization error

---

## Motivation

Feature extraction is a fundamental component in:

- Visual SLAM
- Structure-from-Motion (SfM)
- Autonomous driving perception
- Robotics navigation systems

---

## Project Structure

```
feature-extraction-pipeline/
├── src/               # Core implementation
├── evaluation/        # Benchmarking & metrics
├── examples/          # Demo scripts
├── tests/             # Unit tests
├── config/            # Configuration files
├── results/           # Output visualizations
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/feature-extraction-pipeline.git
cd feature-extraction-pipeline

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

---

## Quick Start

```bash
python examples/demo.py
```

---

## Benchmark

```bash
python -m evaluation.benchmark --config config/config.yaml
```

---

## Metrics

| Metric | Description |
|---|---|
| Matching Accuracy | Ratio of correct matches to total matches |
| Precision / Recall / F1 | Standard classification metrics |
| Repeatability | Keypoint consistency across transformed images |
| Localization Error | Pixel-level keypoint displacement |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| OpenCV | Feature detection & matching |
| NumPy | Numerical operations |
| SciPy | Scientific computing |
| Matplotlib | Visualization |
| PyYAML | Configuration management |
| PyTest | Unit testing |

---

## Author

**Malavika Priyesh**

- GitHub: [github.com/yourusername](https://github.com/MalavikaPriyesh)
- LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/malavikapriyesh)

---

## License

This project is licensed under the [MIT License](LICENSE).
