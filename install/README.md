# Installation Files

This directory contains alternative installation files for convenience.

### Recommended Method
The primary way to install dependencies is directly from the root directory using the package configuration:

```bash
# Run this from the project root
pip install -e .
```

### Alternative Methods

These files are provided for those who prefer traditional dependency management:

- `requirements.txt`: For standard pip installation (`pip install -r install/requirements.txt`)
- `environment.yml`: For Conda users (`conda env create -f install/environment.yml`)

**Note:** The source of truth for all dependencies is the `pyproject.toml` file in the root directory. These files are updated to match that specification, but they can cause conflicting behaviour.
