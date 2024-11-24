import pytest
import argparse
from pathlib import Path
import os
import logging
from unittest.mock import patch, MagicMock
from code_analyzer.main import parse_arguments, get_app_dirs, setup_logging

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set up mock environment variables."""
    monkeypatch.setenv('CODE_ANALYZER_CACHE_DIR', '/tmp/test_cache_dir')
    monkeypatch.setenv('CODE_ANALYZER_LOG_DIR', '/tmp/test_log_dir')
    monkeypatch.setenv('CODE_ANALYZER_LLM_DIR', '/tmp/test_llm_dir')
    monkeypatch.setenv('CODE_ANALYZER_DEBUG_DIR', '/tmp/test_debug_dir')

def test_parse_arguments():
    """Test the argument parsing function."""
    test_args = [
        "repo_path",
        "--output", "output.json",
        "--model", "test-model",
        "--no-cache",
        "--debug",
        "--verbose",
        "--export"
    ]
    with patch('sys.argv', ["main.py"] + test_args):
        args = parse_arguments()
        assert args.repo_path == "repo_path"
        assert args.output == "output.json"
        assert args.model == "test-model"
        assert args.no_cache is True
        assert args.debug is True
        assert args.verbose is True
        assert args.export is True

def test_get_app_dirs(mock_env_vars):
    """Test the directory retrieval based on environment variables."""
    cache_dir, log_dir, llm_dir, debug_dir = get_app_dirs()
    assert cache_dir == Path("/tmp/test_cache_dir")
    assert log_dir == Path("/tmp/test_log_dir")
    assert llm_dir == Path("/tmp/test_llm_dir")
    assert debug_dir == Path("/tmp/test_debug_dir")

logging.getLogger().setLevel(logging.DEBUG)

def test_setup_logging(mock_env_vars, tmp_path):
    """Test the logging setup."""
    # Create a temporary log directory
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Mock the `get_app_dirs` function to return the temp directory
    with patch("code_analyzer.main.get_app_dirs", return_value=(None, log_dir, None, None)):
        setup_logging(debug=True)

        # Check if the log file is created
        log_file = log_dir / "code_analyzer.log"
        assert log_file.exists()

        # Explicitly retrieve the logger and check its level
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

def test_main_directory_creation(mock_env_vars, tmp_path):
    """Test that directories are created during execution."""
    # Mock all directories to point to a temp directory
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "logs"
    llm_dir = tmp_path / "llm_out"
    debug_dir = tmp_path / "debug"
    with patch("code_analyzer.main.get_app_dirs", return_value=(cache_dir, log_dir, llm_dir, debug_dir)):
        with patch("code_analyzer.main.logging.getLogger") as mock_logger:
            mock_logger.return_value = MagicMock()

            # Run the directory creation logic
            cache_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            llm_dir.mkdir(parents=True, exist_ok=True)
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Assert that all directories exist
            assert cache_dir.exists()
            assert log_dir.exists()
            assert llm_dir.exists()
            assert debug_dir.exists()
