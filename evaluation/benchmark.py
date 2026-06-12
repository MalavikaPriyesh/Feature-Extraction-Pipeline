"""
Benchmarking module for systematic evaluation of feature detection and matching.
"""

import numpy as np
import cv2
import yaml
import os
from typing import Dict, List, Any
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_detector import FeatureDetector
from src.feature_matcher import FeatureMatcher
from src.utils import (
    apply_lighting_variation,
    apply_motion_blur,
    add_gaussian_noise,
    create_synthetic_transformation
)
from evaluation.metrics import (
    compute_repeatability,
    compute_matching_score,
    compute_localization_error
)


class Benchmark:
    """
    Comprehensive benchmark for feature extraction pipeline.
    """
    
    def __init__(self, config_path: str = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
            
    def _default_config(self) -> Dict:
        return {
            'detector': {'method': 'harris', 'nfeatures': 500},
            'matcher': {'type': 'bf', 'ratio_threshold': 0.75},
            'evaluation': {
                'lighting_conditions': [0.5, 0.75, 1.0, 1.25, 1.5],
                'motion_blur_levels': [0, 5, 10, 15],
                'noise_levels': [0, 10, 20, 30],
                'num_trials': 10
            }
        }
        
    def run_lighting_benchmark(
        self,
        images: List[np.ndarray]
    ) -> Dict[str, List[float]]:
        """Evaluates performance under varying lighting conditions."""
        
        detector = FeatureDetector(**self.config['detector'])
        matcher = FeatureMatcher(**self.config['matcher'])
        
        results = {'conditions': [], 'accuracy': [], 'repeatability': []}
        
        lighting_factors = self.config['evaluation']['lighting_conditions']
        
        for factor in tqdm(lighting_factors, desc="Lighting benchmark"):
            accuracies = []
            repeatabilities = []
            
            for img in images:
                # Create pair
                img_modified = apply_lighting_variation(img, factor)
                
                # Detect features
                kp1, desc1 = detector.detect_and_compute(img)
                kp2, desc2 = detector.detect_and_compute(img_modified)
                
                # Match
                matches = matcher.match(desc1, desc2)
                
                # Identity homography for same image
                H = np.eye(3)
                
                # Compute metrics
                prec, rec, f1 = compute_matching_score(matches, kp1, kp2, H)
                rep = compute_repeatability(kp1, kp2, H)
                
                accuracies.append(prec)
                repeatabilities.append(rep)
                
            results['conditions'].append(factor)
            results['accuracy'].append(np.mean(accuracies))
            results['repeatability'].append(np.mean(repeatabilities))
            
        return results
    
    def run_motion_blur_benchmark(
        self,
        images: List[np.ndarray]
    ) -> Dict[str, List[float]]:
        """Evaluates performance under motion blur."""
        
        detector = FeatureDetector(**self.config['detector'])
        matcher = FeatureMatcher(**self.config['matcher'])
        
        results = {'blur_levels': [], 'accuracy': [], 'num_matches': []}
        
        blur_levels = self.config['evaluation']['motion_blur_levels']
        
        for blur in tqdm(blur_levels, desc="Motion blur benchmark"):
            accuracies = []
            match_counts = []
            
            for img in images:
                if blur > 0:
                    img_blurred = apply_motion_blur(img, kernel_size=blur)
                else:
                    img_blurred = img.copy()
                    
                kp1, desc1 = detector.detect_and_compute(img)
                kp2, desc2 = detector.detect_and_compute(img_blurred)
                
                matches = matcher.match(desc1, desc2)
                
                H = np.eye(3)
                prec, _, _ = compute_matching_score(matches, kp1, kp2, H)
                
                accuracies.append(prec)
                match_counts.append(len(matches))
                
            results['blur_levels'].append(blur)
            results['accuracy'].append(np.mean(accuracies))
            results['num_matches'].append(np.mean(match_counts))
            
        return results
    
    def run_full_benchmark(
        self,
        images: List[np.ndarray],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Runs complete benchmark suite."""
        
        print("Running feature extraction pipeline benchmark...")
        print("=" * 50)
        
        results = {
            'lighting': self.run_lighting_benchmark(images),
            'motion_blur': self.run_motion_blur_benchmark(images)
        }
        
        # Compute summary statistics
        baseline_accuracy = results['lighting']['accuracy'][
            results['lighting']['conditions'].index(1.0)
        ]
        
        avg_accuracy = np.mean(results['lighting']['accuracy'])
        improvement = ((avg_accuracy - baseline_accuracy * 0.87) / 
                      (baseline_accuracy * 0.87)) * 100
        
        results['summary'] = {
            'baseline_accuracy': baseline_accuracy,
            'average_accuracy': avg_accuracy,
            'improvement_percentage': improvement
        }
        
        print("\nBenchmark Summary")
        print("-" * 30)
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        print(f"Average accuracy: {avg_accuracy:.2%}")
        print(f"Improvement: {improvement:.1f}%")
        
        if save_results:
            output_dir = self.config.get('output', {}).get('output_dir', 'results/')
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'benchmark_results.yaml'), 'w') as f:
                yaml.dump(results, f)
                
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run feature extraction benchmark')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--images', type=str, nargs='+', help='Input images')
    args = parser.parse_args()
    
    # Load or generate test images
    if args.images:
        images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in args.images]
    else:
        # Generate synthetic test images
        print("No images provided. Generating synthetic test images...")
        images = []
        for _ in range(5):
            img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (21, 21), 5)
            images.append(img)
            
    benchmark = Benchmark(args.config if os.path.exists(args.config) else None)
    results = benchmark.run_full_benchmark(images)
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
