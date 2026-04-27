"""Shared test configuration and fixtures"""

import pytest
from rich.console import Console

# Test configuration
PROJECT_ID = "ytpuq"
FILE_PATH = "rpp_data.csv"


@pytest.fixture
def console():
    """Provide a Rich console for tests"""
    return Console()


@pytest.fixture
def output_dir(tmp_path):
    """Create and provide an isolated output directory for test files"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def project_id():
    """Provide the test project ID"""
    return PROJECT_ID


@pytest.fixture
def file_path():
    """Provide the test file path"""
    return FILE_PATH
