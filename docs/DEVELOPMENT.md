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
