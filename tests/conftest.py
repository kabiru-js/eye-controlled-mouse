"""Pytest configuration and fixtures."""
import pytest

@pytest.fixture
def sample_image():
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)
