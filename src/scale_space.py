"""
Scale-space construction module for multi-scale feature detection.
Implements Gaussian pyramid and Difference of Gaussians (DoG).
"""

import numpy as np
import cv2
from typing import List, Tuple


class ScaleSpace:
    """
    Constructs scale-space representation of an image using Gaussian pyramids.
    
    Attributes:
        num_octaves: Number of octaves in the pyramid
        num_scales: Number of scales per octave
        sigma_base: Base sigma for Gaussian blur
        k: Scale factor between consecutive scales
    """
    
    def __init__(
        self,
        num_octaves: int = 4,
        num_scales: int = 5,
        sigma_base: float = 1.6,
        k: float = None
    ):
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma_base = sigma_base
        self.k = k if k else 2 ** (1.0 / num_scales)
        
    def build_gaussian_pyramid(self, image: np.ndarray) -> List[List[np.ndarray]]:
        """
        Builds Gaussian pyramid with multiple octaves and scales.
        
        Args:
            image: Input grayscale image (H, W)
            
        Returns:
            List of octaves, each containing list of blurred images
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        image = image.astype(np.float64)
        pyramid = []
        
        for octave in range(self.num_octaves):
            octave_images = []
            
            for scale in range(self.num_scales + 3):
                sigma = self.sigma_base * (self.k ** scale)
                kernel_size = int(6 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                    
                blurred = cv2.GaussianBlur(
                    image, 
                    (kernel_size, kernel_size), 
                    sigma
                )
                octave_images.append(blurred)
                
            pyramid.append(octave_images)
            
            # Downsample for next octave
            image = cv2.resize(
                octave_images[self.num_scales],
                (image.shape[1] // 2, image.shape[0] // 2),
                interpolation=cv2.INTER_NEAREST
            )
            
        return pyramid
    
    def build_dog_pyramid(self, image: np.ndarray) -> List[List[np.ndarray]]:
        """
        Builds Difference of Gaussians (DoG) pyramid.
        
        Args:
            image: Input grayscale image
            
        Returns:
            List of octaves, each containing DoG images
        """
        gaussian_pyramid = self.build_gaussian_pyramid(image)
        dog_pyramid = []
        
        for octave_images in gaussian_pyramid:
            dog_octave = []
            for i in range(len(octave_images) - 1):
                dog = octave_images[i + 1] - octave_images[i]
                dog_octave.append(dog)
            dog_pyramid.append(dog_octave)
            
        return dog_pyramid
    
    def get_scale_at_level(self, octave: int, scale: int) -> float:
        """Returns the sigma value at a given octave and scale level."""
        return self.sigma_base * (2 ** octave) * (self.k ** scale)
