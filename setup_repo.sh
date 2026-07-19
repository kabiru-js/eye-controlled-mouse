#!/bin/bash
# Repository Restructuring Setup Script
# This script creates all necessary files and directories for the eye-controlled-mouse project

set -e

echo "🚀 Starting Eye-Controlled Mouse Repository Restructuring..."
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p src/tracking
mkdir -p src/control
mkdir -p src/utils
mkdir -p tests
mkdir -p docs
mkdir -p .github/workflows
mkdir -p config

echo "✅ Directories created"
echo ""

# Create pyproject.toml
echo "📝 Creating pyproject.toml..."
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=65.0"]
build-backend = "setuptools.build_meta"

[project]
name = "eye-controlled-mouse"
version = "0.1.0"
description = "A desktop application to control the mouse cursor using your eyes, built with Python."
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Kabiru JS", email = "your.email@example.com"}
]
keywords = ["eye-tracking", "gaze-estimation", "computer-vision", "accessibility"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: System :: Hardware :: Hardware Drivers",
]

dependencies = [
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "pyautogui>=0.9.53",
    "mediapipe>=0.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "ruff>=0.0.280",
    "mypy>=1.4.0",
    "bandit>=1.7.5",
]

[project.urls]
Homepage = "https://github.com/kabiru-js/eye-controlled-mouse"
Repository = "https://github.com/kabiru-js/eye-controlled-mouse.git"
Issues = "https://github.com/kabiru-js/eye-controlled-mouse/issues"

[tool.black]
line-length = 100
target-version = ["py39"]
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
  | tests/.*?/setup.py
)/
'''

[tool.ruff]
line-length = 100
target-version = "py39"
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "C",    # flake8-comprehensions
    "B",    # flake8-bugbear
    "UP",   # pyupgrade
]
ignore = [
    "E501",  # line too long (black handles this)
    "C901",  # function is too complex
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "--cov=src --cov-report=html --cov-report=term-missing"
filterwarnings = [
    "ignore::DeprecationWarning",
]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true
EOF
echo "✅ pyproject.toml created"
echo ""

# Create requirements.txt
echo "📝 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core dependencies
opencv-python>=4.8.0
numpy>=1.24.0
pyautogui>=0.9.53
mediapipe>=0.10.0
EOF
echo "✅ requirements.txt created"
echo ""

# Create requirements-dev.txt
echo "📝 Creating requirements-dev.txt..."
cat > requirements-dev.txt << 'EOF'
# Development and testing dependencies
-r requirements.txt

pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
ruff>=0.0.280
mypy>=1.4.0
bandit>=1.7.5
EOF
echo "✅ requirements-dev.txt created"
echo ""

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDE settings
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Project-specific
config/local_*.json
*.pyc
.webcam_cache/
EOF
echo "✅ .gitignore created"
echo ""

# Create package __init__.py files
echo "📝 Creating package initialization files..."

cat > src/__init__.py << 'EOF'
"""Eye-Controlled Mouse - Control your cursor with your eyes."""

__version__ = "0.1.0"
EOF

cat > src/tracking/__init__.py << 'EOF'
"""Eye tracking and gaze estimation module."""

from .eye_detection import EyeDetector
from .blink_detection import BlinkDetector
from .gaze_mapping import GazeMapper

__all__ = ["EyeDetector", "BlinkDetector", "GazeMapper"]
EOF

cat > src/control/__init__.py << 'EOF'
"""Cursor control and OS integration module."""

from .cursor_controller import CursorController

__all__ = ["CursorController"]
EOF

cat > src/utils/__init__.py << 'EOF'
"""Utility functions and configuration management."""

from .config_loader import ConfigLoader

__all__ = ["ConfigLoader"]
EOF

cat > tests/__init__.py << 'EOF'
"""Test suite for eye-controlled-mouse."""
EOF

echo "✅ Package initialization files created"
echo ""

# Create pytest configuration
echo "📝 Creating pytest configuration..."
cat > tests/conftest.py << 'EOF'
"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_image():
    """Fixture to provide sample image data for testing."""
    import numpy as np

    # Create a dummy image (480x640x3 BGR format)
    return np.zeros((480, 640, 3), dtype=np.uint8)
EOF

cat > tests/test_example.py << 'EOF'
"""Example test file to verify pytest setup."""


def test_imports():
    """Test that core modules can be imported."""
    from src.tracking import EyeDetector, BlinkDetector, GazeMapper
    from src.control import CursorController
    from src.utils import ConfigLoader

    assert EyeDetector is not None
    assert BlinkDetector is not None
    assert GazeMapper is not None
    assert CursorController is not None
    assert ConfigLoader is not None


def test_example():
    """Simple test to verify pytest works."""
    assert 1 + 1 == 2
EOF

echo "✅ Pytest configuration created"
echo ""

# Create GitHub Actions workflow
echo "📝 Creating GitHub Actions CI/CD pipeline..."
cat > .github/workflows/ci.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    name: Linting & Code Quality

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Check code formatting with black
        run: black --check src tests

      - name: Lint with ruff
        run: ruff check src tests

      - name: Type check with mypy
        run: mypy src
        continue-on-error: true

  test:
    runs-on: ubuntu-latest
    name: Tests (Python ${{ matrix.python-version }})
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests
          fail_ci_if_error: false

  security:
    runs-on: ubuntu-latest
    name: Security Checks

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Run bandit security check
        run: |
          pip install bandit
          bandit -r src -ll
        continue-on-error: true
EOF

echo "✅ GitHub Actions workflow created"
echo ""

# Create DEVELOPMENT.md
echo "📝 Creating DEVELOPMENT.md guide..."
cat > docs/DEVELOPMENT.md << 'EOF'
# Development Guide

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/kabiru-js/eye-controlled-mouse.git
cd eye-controlled-mouse
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
# For development (includes testing & linting tools)
pip install -r requirements-dev.txt

# Or use pyproject.toml
pip install -e ".[dev]"
```

## Code Quality

### Formatting
We use **Black** for consistent code formatting.
```bash
black src tests
```

### Linting
We use **Ruff** for fast linting and code analysis.
```bash
ruff check src tests
```

### Type Checking
Optional type checking with **mypy**:
```bash
mypy src
```

### Run All Quality Checks
```bash
# Format
black src tests

# Lint
ruff check src tests

# Test with coverage
pytest --cov=src
```

## Testing

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_example.py
```

### Run with verbose output
```bash
pytest -v
```

## Project Structure

```
eye-controlled-mouse/
├── src/                      # Source code
│   ├── main.py              # Entry point
│   ├── tracking/            # Eye tracking module
│   │   ├── eye_detection.py
│   │   ├── blink_detection.py
│   │   └── gaze_mapping.py
│   ├── control/             # Cursor control module
│   │   └── cursor_controller.py
│   └── utils/               # Utilities
│       └── config_loader.py
├── tests/                    # Test suite
│   ├── test_example.py
│   └── conftest.py
├── config/                   # Configuration files
│   └── config.json
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md
│   ├── DECISIONS.md
│   └── DEVELOPMENT.md
├── .github/workflows/        # CI/CD
│   └── ci.yml
├── pyproject.toml           # Modern Python packaging
├── requirements.txt         # Core dependencies
├── requirements-dev.txt     # Development dependencies
├── README.md
└── .gitignore
```

## Creating a Pull Request

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and commit: `git commit -m "feat: your feature description"`
3. Run quality checks: `black src tests && ruff check src tests && pytest`
4. Push to GitHub: `git push origin feature/your-feature`
5. Open a Pull Request

### Commit Message Format
Follow conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for code refactoring
- `chore:` for maintenance

Example: `feat: implement gaze mapping calibration`

## Troubleshooting

### Import errors
Make sure you're in the virtual environment and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Tests not discovering
Ensure test files follow naming convention: `test_*.py` or `*_test.py`

### Black/Ruff conflicts
Run black first, then ruff:
```bash
black src tests
ruff check src tests --fix
```

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
- [pyproject.toml Specification](https://www.python-poetry.org/docs/pyproject/)
EOF

echo "✅ DEVELOPMENT.md created"
echo ""

# Create enhanced README.md
echo "📝 Creating enhanced README.md..."
cat > README.md << 'EOF'
# Eye-Controlled Cursor System

> **Control your computer with just your eyes.** A real-time computer vision system that enables hands-free mouse control using eye movements and blink detection via a standard webcam.

[![CI/CD Pipeline](https://github.com/kabiru-js/eye-controlled-mouse/actions/workflows/ci.yml/badge.svg)](https://github.com/kabiru-js/eye-controlled-mouse/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 The Problem

Traditional input devices (mouse, trackpad) create barriers for people with motor impairments. They're also inefficient when your hands are full or in sterile environments.

**This project reimagines human-computer interaction** through gaze tracking, proving that your eyes alone can be a powerful input device.

---

## ✨ How It Works

The system processes webcam input in real time:

1. **👁️ Face & Eye Detection**
   - Real-time facial landmark detection using computer vision
   - Precise eye region isolation

2. **🎯 Gaze Estimation**
   - Maps eye position to screen coordinates
   - Translates gaze into smooth cursor movement

3. **👀 Blink Detection**
   - **Both eyes closed** → Left click
   - **Right eye wink** → Right click
   - **Rapid blinking** → Double click

4. **🖱️ Cursor Control**
   - OS-level mouse events based on interpreted signals
   - Sub-pixel accurate positioning

---

## 🏗️ System Architecture

```
┌──────────────────┐
│ Webcam Input     │
└────────┬─────────┘
         ↓
┌──────────────────────────────────┐
│ Face & Eye Detection (MediaPipe) │
└────────┬─────────────────────────┘
         ↓
┌──────────────────────────────────┐
│ Gaze Mapping (Eye→Screen)        │
└────────┬─────────────────────────┘
         ↓
┌──────────────────────────────────┐
│ Blink Detection (Event Recog.)   │
└────────┬─────────────────────────┘
         ↓
┌──────────────────────────────────┐
│ Cursor Controller (OS Events)    │
└────────┬─────────────────────────┘
         ↓
┌──────────────────┐
│ Mouse Cursor     │
└──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Webcam
- macOS / Windows / Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/kabiru-js/eye-controlled-mouse.git
cd eye-controlled-mouse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python src/main.py
```

**Controls:**
- Move eyes to control cursor
- Blink once to left-click
- Wink right eye to right-click
- Rapid blink for double-click
- Press `ESC` to exit

---

## 🛠️ Development

For detailed development setup and guidelines, see [DEVELOPMENT.md](docs/DEVELOPMENT.md).

### Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

### Code Quality

```bash
# Format code
black src tests

# Lint with ruff
ruff check src tests

# Run tests with coverage
pytest --cov=src
```

### Run Tests

```bash
pytest
```

---

## 📊 Project Structure

```
eye-controlled-mouse/
├── src/
│   ├── main.py                 # Entry point
│   ├── tracking/               # Eye tracking module
│   │   ├── eye_detection.py    # Face & eye detection
│   │   ├── blink_detection.py  # Blink recognition
│   │   └── gaze_mapping.py     # Eye→screen mapping
│   ├── control/                # Cursor control
│   │   └── cursor_controller.py
│   └── utils/
│       └── config_loader.py
├── tests/                      # Test suite
├── config/                     # Configuration
│   └── config.json
├── docs/                       # Documentation
├── .github/workflows/          # CI/CD
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Dependencies
└── README.md
```

---

## 🔧 Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| **Computer Vision** | MediaPipe | Fast, accurate, no training needed |
| **Image Processing** | OpenCV | Industry standard, mature, battle-tested |
| **Cursor Control** | PyAutoGUI | Simple, cross-platform |
| **Numerics** | NumPy | Efficient matrix operations |
| **Testing** | pytest | Modern, simple, powerful |
| **Linting** | ruff | Fast, modern alternative to flake8 |
| **Formatting** | black | Consistent, zero-config formatting |
| **CI/CD** | GitHub Actions | Native, reliable, no additional cost |

---

## 📖 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - Technical design and system components
- **[Decisions](docs/DECISIONS.md)** - Engineering decisions and tradeoffs
- **[Development Guide](docs/DEVELOPMENT.md)** - Setup and contribution guidelines

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`)
4. **Run tests and linting** (see [DEVELOPMENT.md](docs/DEVELOPMENT.md))
5. **Push** to your fork (`git push origin feature/amazing-feature`)
6. **Open a Pull Request**

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed development guidelines.

---

## 🐛 Known Limitations

- **Lighting:** Requires adequate lighting for reliable eye detection
- **Glasses:** Some reflective glasses may interfere with detection
- **Calibration:** Currently requires manual position adjustment
- **Latency:** ~100-150ms depending on system specs

---

## 🗺️ Roadmap

- [ ] Calibration wizard for first-time setup
- [ ] Configurable sensitivity profiles
- [ ] Eye gaze zone system (for enhanced precision)
- [ ] GUI application (PyQt)
- [ ] Multi-monitor support
- [ ] Data collection for ML improvements
- [ ] Desktop installer (Windows/macOS/Linux)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Inspiration

Built for accessibility, powered by computer vision, enabled by open source.

If this project helps you or sparks ideas, consider giving it a ⭐!

---

## 📬 Questions?

- **Issues:** [GitHub Issues](https://github.com/kabiru-js/eye-controlled-mouse/issues)
- **Discussions:** [GitHub Discussions](https://github.com/kabiru-js/eye-controlled-mouse/discussions)

---

**Made with 👁️ by [Kabiru JS](https://github.com/kabiru-js)**
EOF

echo "✅ README.md created"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Repository restructuring complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Files Created:"
echo "  ✓ pyproject.toml (modern Python packaging)"
echo "  ✓ requirements.txt (core dependencies)"
echo "  ✓ requirements-dev.txt (dev dependencies)"
echo "  ✓ .gitignore (comprehensive)"
echo "  ✓ src/__init__.py through src/utils/__init__.py"
echo "  ✓ tests/__init__.py, conftest.py, test_example.py"
echo "  ✓ .github/workflows/ci.yml (GitHub Actions)"
echo "  ✓ docs/DEVELOPMENT.md (dev guide)"
echo "  ✓ README.md (enhanced)"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review all created files"
echo "  2. Update author email in pyproject.toml"
echo "  3. Add git and commit:"
echo "     git add ."
echo "     git commit -m \"feat: complete repository restructuring with modern Python standards\""
echo "  4. Push to GitHub:"
echo "     git push origin main"
echo ""
echo "🧪 To get started with development:"
echo "  python -m venv venv"
echo "  source venv/bin/activate  # Windows: venv\\Scripts\\activate"
echo "  pip install -r requirements-dev.txt"
echo "  pytest"
echo ""
echo "📚 For more information, see docs/DEVELOPMENT.md"
echo ""
EOF

chmod +x setup_repo.sh

echo "✅ Script created successfully!"
