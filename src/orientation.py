"""
Orientation assignment module for rotation-invariant feature description.
"""

import numpy as np
import cv2
from typing import List


class OrientationAssignment:
    """
    Assigns dominant orientation to keypoints for rotation invariance.
    """
    
    def __init__(self, num_bins: int = 36, peak_ratio: float = 0.8):
        self.num_bins = num_bins
        self.peak_ratio = peak_ratio
        self.bin_width = 360.0 / num_bins
        
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> List[cv2.KeyPoint]:
        """
        Computes and assigns orientation to each keypoint.
        
        Args:
            image: Grayscale image
            keypoints: List of detected keypoints
            
        Returns:
            Keypoints with assigned orientations
        """
        # Compute gradients
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * 180.0 / np.pi
        orientation[orientation < 0] += 360.0
        
        oriented_keypoints = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            scale = kp.size if kp.size > 0 else 1.0
            
            # Define patch size based on scale
            radius = int(3 * scale)
            
            # Boundary check
            if (x - radius < 0 or x + radius >= image.shape[1] or
                y - radius < 0 or y + radius >= image.shape[0]):
                kp.angle = 0.0
                oriented_keypoints.append(kp)
                continue
                
            # Build orientation histogram
            histogram = np.zeros(self.num_bins)
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    px, py = x + j, y + i
                    
                    # Gaussian weight
                    weight = np.exp(-(i**2 + j**2) / (2 * (1.5 * scale)**2))
                    
                    mag = magnitude[py, px]
                    ori = orientation[py, px]
                    
                    bin_idx = int(ori / self.bin_width) % self.num_bins
                    histogram[bin_idx] += weight * mag
                    
            # Smooth histogram
            histogram = np.convolve(histogram, [1, 4, 6, 4, 1], mode='same') / 16.0
            
            # Find dominant orientation
            max_val = histogram.max()
            max_idx = histogram.argmax()
            
            # Parabolic interpolation for sub-bin accuracy
            left = histogram[(max_idx - 1) % self.num_bins]
            right = histogram[(max_idx + 1) % self.num_bins]
            
            interp_offset = 0.5 * (left - right) / (left - 2 * max_val + right + 1e-8)
            angle = (max_idx + 0.5 + interp_offset) * self.bin_width
            angle = angle % 360.0
            
            kp.angle = angle
            oriented_keypoints.append(kp)
            
            # Create additional keypoints for secondary peaks
            for i, val in enumerate(histogram):
                if i != max_idx and val >= self.peak_ratio * max_val:
                    new_kp = cv2.KeyPoint(
                        x=kp.pt[0],
                        y=kp.pt[1],
                        size=kp.size,
                        angle=(i + 0.5) * self.bin_width,
                        response=kp.response,
                        octave=kp.octave
                    )
                    oriented_keypoints.append(new_kp)
                    
        return oriented_keypoints
