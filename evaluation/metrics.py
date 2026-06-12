"""
Evaluation metrics for feature detection and matching.
"""

import numpy as np
from typing import List, Tuple
import cv2


def compute_repeatability(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    homography: np.ndarray,
    threshold: float = 3.0
) -> float:
    """
    Computes repeatability score between two sets of keypoints.
    
    Args:
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        homography: Ground truth homography from image 1 to 2
        threshold: Distance threshold for correspondence
        
    Returns:
        Repeatability score (0 to 1)
    """
    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0
        
    # Transform kp1 to image 2 coordinates
    pts1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts1_transformed = (homography @ pts1_h.T).T
    pts1_transformed = pts1_transformed[:, :2] / pts1_transformed[:, 2:3]
    
    pts2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2])
    
    # Count correspondences
    correspondences = 0
    for pt1 in pts1_transformed:
        distances = np.linalg.norm(pts2 - pt1, axis=1)
        if distances.min() < threshold:
            correspondences += 1
            
    repeatability = correspondences / min(len(kp1), len(kp2))
    
    return repeatability


def compute_matching_score(
    matches: List[cv2.DMatch],
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    homography: np.ndarray,
    threshold: float = 3.0
) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F1 score for matches.
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    if len(matches) == 0:
        return 0.0, 0.0, 0.0
        
    # Count correct matches
    correct = 0
    for m in matches:
        pt1 = np.array([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], 1.0])
        pt2_actual = np.array(kp2[m.trainIdx].pt)
        
        pt1_transformed = homography @ pt1
        pt1_transformed = pt1_transformed[:2] / pt1_transformed[2]
        
        if np.linalg.norm(pt1_transformed - pt2_actual) < threshold:
            correct += 1
            
    # Compute metrics
    precision = correct / len(matches) if len(matches) > 0 else 0
    
    # Estimate total possible matches
    total_possible = min(len(kp1), len(kp2))
    recall = correct / total_possible if total_possible > 0 else 0
    
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1


def compute_localization_error(
    matches: List[cv2.DMatch],
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    homography: np.ndarray
) -> float:
    """
    Computes mean localization error for matches.
    
    Returns:
        Mean error in pixels
    """
    if len(matches) == 0:
        return float('inf')
        
    errors = []
    for m in matches:
        pt1 = np.array([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], 1.0])
        pt2_actual = np.array(kp2[m.trainIdx].pt)
        
        pt1_transformed = homography @ pt1
        pt1_transformed = pt1_transformed[:2] / pt1_transformed[2]
        
        error = np.linalg.norm(pt1_transformed - pt2_actual)
        errors.append(error)
        
    return np.mean(errors)
