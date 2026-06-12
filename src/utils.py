"""
Utility functions for visualization and data processing.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def draw_keypoints(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    color: Tuple[int, int, int] = (0, 255, 0),
    draw_orientation: bool = True
) -> np.ndarray:
    """Draws keypoints on image with optional orientation arrows."""
    
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
        
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_orientation else 0
    
    output = cv2.drawKeypoints(
        output, keypoints, None, color=color, flags=flags
    )
    
    return output


def draw_matches(
    img1: np.ndarray,
    kp1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    max_matches: int = 50
) -> np.ndarray:
    """Draws feature matches between two images."""
    
    matches_to_draw = matches[:max_matches]
    
    output = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return output


def apply_lighting_variation(
    image: np.ndarray,
    factor: float
) -> np.ndarray:
    """Applies brightness variation to image."""
    
    adjusted = image.astype(np.float32) * factor
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int = 15,
    angle: float = 0
) -> np.ndarray:
    """Applies directional motion blur."""
    
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    
    # Rotate kernel
    M = cv2.getRotationMatrix2D(
        (kernel_size // 2, kernel_size // 2), angle, 1.0
    )
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    
    blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 25.0
) -> np.ndarray:
    """Adds Gaussian noise to image."""
    
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy


def create_synthetic_transformation(
    image: np.ndarray,
    rotation: float = 0,
    scale: float = 1.0,
    translation: Tuple[float, float] = (0, 0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates synthetically transformed image with ground truth homography.
    
    Returns:
        Tuple of (transformed_image, homography_matrix)
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Build transformation matrix
    M = cv2.getRotationMatrix2D(center, rotation, scale)
    M[0, 2] += translation[0]
    M[1, 2] += translation[1]
    
    # Convert to 3x3 homography
    H = np.vstack([M, [0, 0, 1]])
    
    transformed = cv2.warpAffine(image, M, (w, h))
    
    return transformed, H
