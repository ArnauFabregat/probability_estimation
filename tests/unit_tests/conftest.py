import sys
from pathlib import Path

import pytest

# Define the absolute path to the root of the project directory
PROJECT_ROOT_DIR = Path.cwd().resolve().parents[2]

# Define the paths to the source, tests and data directories
SRC_DIR = PROJECT_ROOT_DIR / "src"

# Add the source directory to the system path for module imports
sys.path.append(str(SRC_DIR))


@pytest.fixture
def example_config() -> dict[str, str]:
    params = {
        "pipeline_name": "feature_acquisition_pipeline",
        "dag_run_id": "0",
        "country_code": "US",
        "environment_tag": "dev",
    }
    return params
