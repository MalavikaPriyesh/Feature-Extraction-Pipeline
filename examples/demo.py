"""
Demo script showing feature extraction pipeline usage.
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_detector import FeatureDetector
from src.feature_matcher import FeatureMatcher
from src.utils import draw_keypoints, draw_matches, create_synthetic_transformation


def main():
    print("Feature Extraction Pipeline Demo")
    print("=" * 40)
    
    # Create or load test image
    print("\nGenerating test image...")
    
    # Create synthetic test image with features
    img = np.zeros((400, 600), dtype=np.uint8)
    
    # Add some geometric shapes
    cv2.rectangle(img, (50, 50), (150, 150), 255, 2)
    cv2.rectangle(img, (200, 80), (280, 180), 200, 2)
    cv2.circle(img, (400, 100), 50, 255, 2)
    cv2.line(img, (100, 300), (500, 350), 200, 2)
    
    # Add noise for texture
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 1)
    
    # Create transformed version
    print("Creating transformed image...")
    img_transformed, H = create_synthetic_transformation(
        img, rotation=15, scale=0.9, translation=(20, 10)
    )
    
    # Initialize detector and matcher
    print("\nInitializing feature detector (Harris)...")
    detector = FeatureDetector(method='harris', nfeatures=200)
    matcher = FeatureMatcher(matcher_type='bf', ratio_threshold=0.75)
    
    # Detect features
    print("Detecting features...")
    kp1, desc1 = detector.detect_and_compute(img)
    kp2, desc2 = detector.detect_and_compute(img_transformed)
    
    print(f"  Image 1: {len(kp1)} keypoints")
    print(f"  Image 2: {len(kp2)} keypoints")
    
    # Match features
    print("\nMatching features...")
    matches = matcher.match(desc1, desc2)
    print(f"  Found {len(matches)} matches")
    
    # Geometric verification
    print("\nApplying geometric verification (RANSAC)...")
    filtered_matches, H_est = matcher.match_and_filter(kp1, desc1, kp2, desc2)
    print(f"  Inlier matches: {len(filtered_matches)}")
    
    # Compute accuracy
    if len(filtered_matches) > 0:
        accuracy = FeatureMatcher.compute_matching_accuracy(
            filtered_matches, kp1, kp2, H, threshold=5.0
        )
        print(f"  Matching accuracy: {accuracy:.2%}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    img_kp1 = draw_keypoints(img, kp1)
    img_kp2 = draw_keypoints(img_transformed, kp2)
    img_matches = draw_matches(img, kp1, img_transformed, kp2, filtered_matches)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    cv2.imwrite('results/keypoints_1.png', img_kp1)
    cv2.imwrite('results/keypoints_2.png', img_kp2)
    cv2.imwrite('results/matches.png', img_matches)
    
    print("\nResults saved to 'results/' directory")
    print("Demo complete!")


if __name__ == '__main__':
    main()
