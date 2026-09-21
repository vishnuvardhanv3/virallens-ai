"""
Pytest configuration for ViralLens test suite.
Pre-loads PyTorch DLLs on Windows to avoid OpenMP library collisions with OpenCV / scikit-learn.
"""
try:
    import torch
except Exception:
    pass
