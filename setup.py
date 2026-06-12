from setuptools import setup, find_packages

setup(
    name='feature-extraction-pipeline',
    version='1.0.0',
    author='Malavika Priyesh',
    author_email='malavikapriyesh@gmail.com',
    description='ML-based feature extraction pipeline for autonomous driving',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'opencv-contrib-python>=4.5.0',
        'matplotlib>=3.4.0',
        'scipy>=1.7.0',
        'pyyaml>=5.4.0',
        'tqdm>=4.62.0',
    ],
    extras_require={
        'dev': ['pytest>=6.2.0'],
    },
)
