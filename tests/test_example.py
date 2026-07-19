"""Example test file to verify pytest setup."""

def test_imports():
    from src.tracking import EyeDetector, BlinkDetector, GazeMapper
    from src.control import CursorController
    from src.utils import ConfigLoader
    assert EyeDetector is not None
    assert BlinkDetector is not None
    assert GazeMapper is not None
    assert CursorController is not None
    assert ConfigLoader is not None

def test_example():
    assert 1 + 1 == 2
