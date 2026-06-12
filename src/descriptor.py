"""
Descriptor generation module for feature matching.
Implements SIFT-like and ORB-like descriptors.
"""

import numpy as np
import cv2
from typing import List, Optional


class DescriptorGenerator:
    """
    Generates feature descriptors for detected keypoints.
    """
    
    def __init__(
        self,
        descriptor_type: str = 'sift',
        patch_size: int = 16,
        num_bins: int = 8
    ):
        self.descriptor_type = descriptor_type.lower()
        self.patch_size = patch_size
        self.num_bins = num_bins
        
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> np.ndarray:
        """
        Computes descriptors for keypoints.
        
        Args:
            image: Grayscale image
            keypoints: List of keypoints
            
        Returns:
            Descriptor array of shape (N, descriptor_dim)
        """
        if self.descriptor_type == 'sift':
            return self._compute_sift_descriptors(image, keypoints)
        elif self.descriptor_type == 'orb':
            return self._compute_orb_descriptors(image, keypoints)
        elif self.descriptor_type == 'brief':
            return self._compute_brief_descriptors(image, keypoints)
        else:
            raise ValueError(f"Unknown descriptor type: {self.descriptor_type}")
            
    def _compute_sift_descriptors(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> np.ndarray:
        """Computes SIFT-like descriptors."""
        
        # Compute gradients
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * 180.0 / np.pi
        orientation[orientation < 0] += 360.0
        
        descriptors = []
        
        for kp in keypoints:
            desc = self._compute_single_sift(
                magnitude, orientation, kp
            )
            descriptors.append(desc)
            
        if len(descriptors) == 0:
            return np.array([]).reshape(0, 128)
            
        descriptors = np.array(descriptors, dtype=np.float32)
        
        # Normalize descriptors
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        descriptors = descriptors / norms
        
        # Clip and renormalize
        descriptors = np.clip(descriptors, 0, 0.2)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        descriptors = descriptors / norms
        
        return descriptors
    
    def _compute_single_sift(
        self,
        magnitude: np.ndarray,
        orientation: np.ndarray,
        kp: cv2.KeyPoint
    ) -> np.ndarray:
        """Computes SIFT descriptor for a single keypoint."""
        
        x, y = int(kp.pt[0]), int(kp.pt[1])
        angle = kp.angle if kp.angle >= 0 else 0
        
        # 4x4 grid, 8 orientation bins = 128-dim descriptor
        descriptor = np.zeros((4, 4, 8))
        
        half_size = self.patch_size // 2
        cell_size = self.patch_size // 4
        
        for i in range(-half_size, half_size):
            for j in range(-half_size, half_size):
                px, py = x + j, y + i
                
                # Boundary check
                if (px < 0 or px >= magnitude.shape[1] or
                    py < 0 or py >= magnitude.shape[0]):
                    continue
                    
                # Get magnitude and orientation
                mag = magnitude[py, px]
                ori = orientation[py, px] - angle
                ori = ori % 360.0
                
                # Determine cell
                cell_x = min((j + half_size) // cell_size, 3)
                cell_y = min((i + half_size) // cell_size, 3)
                
                # Determine orientation bin
                bin_idx = int(ori / 45.0) % 8
                
                # Gaussian weight
                weight = np.exp(-(i**2 + j**2) / (2 * (half_size * 0.5)**2))
                
                descriptor[cell_y, cell_x, bin_idx] += weight * mag
                
        return descriptor.flatten()
    
    def _compute_orb_descriptors(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> np.ndarray:
        """Computes ORB-like binary descriptors."""
        
        # Use OpenCV's ORB for actual computation
        orb = cv2.ORB_create(nfeatures=len(keypoints))
        
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        _, descriptors = orb.compute(image_uint8, keypoints)
        
        if descriptors is None:
            return np.array([]).reshape(0, 32)
            
        return descriptors
    
    def _compute_brief_descriptors(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> np.ndarray:
        """Computes BRIEF binary descriptors."""
        
        np.random.seed(42)  # For reproducibility
        
        # Generate random sampling pattern
        patch_size = 31
        n_pairs = 256
        
        pairs = np.random.randint(
            -patch_size // 2,
            patch_size // 2 + 1,
            size=(n_pairs, 4)
        )
        
        # Smooth image
        image = cv2.GaussianBlur(image, (9, 9), 2)
        
        descriptors = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            desc_bits = []
            
            for p in pairs:
                x1, y1, x2, y2 = p
                px1, py1 = x + x1, y + y1
                px2, py2 = x + x2, y + y2
                
                # Boundary check
                if (px1 < 0 or px1 >= image.shape[1] or
                    py1 < 0 or py1 >= image.shape[0] or
                    px2 < 0 or px2 >= image.shape[1] or
                    py2 < 0 or py2 >= image.shape[0]):
                    desc_bits.append(0)
                else:
                    desc_bits.append(1 if image[py1, px1] < image[py2, px2] else 0)
                    
            # Pack bits into bytes
            desc = np.packbits(desc_bits)
            descriptors.append(desc)
            
        if len(descriptors) == 0:
            return np.array([]).reshape(0, 32)
            
        return np.array(descriptors, dtype=np.uint8)
