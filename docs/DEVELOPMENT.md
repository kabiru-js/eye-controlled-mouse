# Development Guide

## Setup

### 1. Clone the repository
\\\ash
git clone https://github.com/kabiru-js/eye-controlled-mouse.git
cd eye-controlled-mouse
\\\

### 2. Create virtual environment
\\\ash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
\\\

### 3. Install dependencies
\\\ash
pip install -r requirements-dev.txt
\\\

## Code Quality

\\\ash
black src tests
ruff check src tests
pytest --cov=src
\\\

## Project Structure

\\\
eye-controlled-mouse/
+-- src/                      # Source code
¦   +-- tracking/            # Eye tracking module
¦   +-- control/             # Cursor control
¦   +-- utils/               # Utilities
+-- tests/                    # Test suite
+-- .github/workflows/        # CI/CD
+-- pyproject.toml
+-- requirements.txt
+-- README.md
\\\

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
