"""
Unit tests for feature detection module.
"""

import pytest
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_detector import FeatureDetector
from src.feature_matcher import FeatureMatcher
from src.scale_space import ScaleSpace


class TestScaleSpace:
    
    def test_gaussian_pyramid_shape(self):
        """Test Gaussian pyramid has correct structure."""
        ss = ScaleSpace(num_octaves=3, num_scales=4)
        img = np.random.rand(256, 256).astype(np.float64)
        
        pyramid = ss.build_gaussian_pyramid(img)
        
        assert len(pyramid) == 3  # num_octaves
        assert len(pyramid[0]) == 7  # num_scales + 3
        
    def test_dog_pyramid_shape(self):
        """Test DoG pyramid has correct structure."""
        ss = ScaleSpace(num_octaves=3, num_scales=4)
        img = np.random.rand(256, 256).astype(np.float64)
        
        dog_pyramid = ss.build_dog_pyramid(img)
        
        assert len(dog_pyramid) == 3
        assert len(dog_pyramid[0]) == 6  # num_scales + 2


class TestFeatureDetector:
    
    @pytest.fixture
    def test_image(self):
        """Creates test image with corners."""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), 255, 2)
        return img
    
    def test_harris_detection(self, test_image):
        """Test Harris corner detection finds corners."""
        detector = FeatureDetector(method='harris', nfeatures=50)
        keypoints = detector.detect(test_image)
        
        assert len(keypoints) > 0
        assert all(isinstance(kp, cv2.KeyPoint) for kp in keypoints)
        
    def test_detect_and_compute(self, test_image):
        """Test combined detection and description."""
        detector = FeatureDetector(method='harris', nfeatures=50)
        keypoints, descriptors = detector.detect_and_compute(test_image)
        
        assert len(keypoints) > 0
        assert descriptors is not None
        assert len(keypoints) == len(descriptors)
        
    def test_nfeatures_limit(self, test_image):
        """Test that nfeatures limits output."""
        detector = FeatureDetector(method='harris', nfeatures=10)
        keypoints = detector.detect(test_image)
        
        assert len(keypoints) <= 10


class TestFeatureMatcher:
    
    def test_bf_matching(self):
        """Test brute-force matching."""
        desc1 = np.random.rand(50, 128).astype(np.float32)
        desc2 = np.random.rand(50, 128).astype(np.float32)
        
        matcher = FeatureMatcher(matcher_type='bf', cross_check=True)
        matches = matcher.match(desc1, desc2)
        
        assert isinstance(matches, list)
        
    def test_empty_descriptors(self):
        """Test handling of empty descriptors."""
        matcher = FeatureMatcher()
        
        matches = matcher.match(np.array([]), np.array([]))
        assert len(matches) == 0
        
        matches = matcher.match(None, None)
        assert len(matches) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

