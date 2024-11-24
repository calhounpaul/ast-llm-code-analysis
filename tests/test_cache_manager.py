import pytest
from datetime import datetime, timedelta
import sqlite3
import json
import base64
from pathlib import Path
from code_analyzer.models.cache_manager import CacheManager, CacheEntry

import sys
sys.path.append('/home/mluser/code_analyzer')


@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture to provide a temporary database path."""
    return str(tmp_path / "test_cache.db")

@pytest.fixture
def cache_manager(temp_db_path):
    """Fixture to provide a CacheManager instance with a temporary database."""
    return CacheManager(temp_db_path)

def test_cache_entry_from_db_row():
    """Test CacheEntry.from_db_row conversion."""
    timestamp = datetime.now()
    metadata = {"test": "data"}
    prompt = "test prompt"
    response = "test response"
    
    # Create encoded test data
    encoded_prompt = base64.b64encode(prompt.encode('utf-8')).decode('utf-8')
    encoded_response = base64.b64encode(response.encode('utf-8')).decode('utf-8')
    
    row = (
        "test_hash",
        encoded_prompt,
        encoded_response,
        timestamp.isoformat(),
        json.dumps(metadata)
    )
    
    entry = CacheEntry.from_db_row(row)
    
    assert entry.query_hash == "test_hash"
    assert entry.prompt == prompt
    assert entry.response == response
    assert entry.timestamp == timestamp
    assert entry.metadata == metadata

def test_cache_basic_operations(cache_manager):
    """Test basic cache operations: put, get, invalidate."""
    prompt = "test prompt"
    response = "test response"
    metadata = {"type": "test"}
    
    # Test put and get
    cache_manager.put(prompt, response, metadata)
    entry = cache_manager.get(prompt)
    
    assert entry is not None
    assert entry.prompt == prompt
    assert entry.response == response
    assert entry.metadata == metadata
    
    # Test invalidate
    assert cache_manager.invalidate(prompt) is True
    assert cache_manager.get(prompt) is None
    assert cache_manager.invalidate(prompt) is False  # Already removed

def test_cache_clear(cache_manager):
    """Test cache clearing functionality."""
    # Add multiple entries
    entries = [
        ("prompt1", "response1"),
        ("prompt2", "response2"),
        ("prompt3", "response3")
    ]
    
    for prompt, response in entries:
        cache_manager.put(prompt, response)
    
    # Test clear all
    assert cache_manager.clear() == len(entries)
    assert cache_manager.get_stats()["total_entries"] == 0

def test_cache_clear_with_timestamp(cache_manager):
    """Test clearing cache entries older than a specified time."""
    # Add entries with different timestamps
    cache_manager.put("old_prompt", "old_response")
    
    # Wait a moment to ensure different timestamps
    middle_time = datetime.now()
    
    cache_manager.put("new_prompt", "new_response")
    
    # Clear entries older than middle_time
    cleared = cache_manager.clear(older_than=middle_time)
    assert cleared == 1
    
    # Verify old entry is gone but new entry remains
    assert cache_manager.get("old_prompt") is None
    assert cache_manager.get("new_prompt") is not None

def test_cache_stats(cache_manager):
    """Test cache statistics functionality."""
    prompt = "test prompt"
    response = "test response"
    
    # Test empty cache stats
    empty_stats = cache_manager.get_stats()
    assert empty_stats["total_entries"] == 0
    assert empty_stats["total_size_bytes"] == 0
    
    # Add an entry and check stats
    cache_manager.put(prompt, response)
    stats = cache_manager.get_stats()
    
    assert stats["total_entries"] == 1
    assert stats["total_size_bytes"] > 0
    assert datetime.fromisoformat(stats["newest_entry"]) <= datetime.now()
    assert stats["oldest_entry"] == stats["newest_entry"]

def test_cache_context_manager(temp_db_path):
    """Test cache manager as context manager."""
    with CacheManager(temp_db_path) as cache:
        cache.put("test", "response")
        assert cache.get("test") is not None

def test_cache_concurrent_connections(temp_db_path):
    """Test multiple cache manager instances accessing same database."""
    cache1 = CacheManager(temp_db_path)
    cache2 = CacheManager(temp_db_path)
    
    cache1.put("test", "response1")
    assert cache2.get("test").response == "response1"
    
    cache2.put("test", "response2")
    assert cache1.get("test").response == "response2"

def test_cache_invalid_data(cache_manager):
    """Test handling of invalid data."""
    with pytest.raises(Exception):
        # Try to store invalid data type
        cache_manager.put("test", {"invalid": "type"})

def test_cache_empty_values(cache_manager):
    """Test handling of empty values."""
    # Test empty prompt
    with pytest.raises(ValueError, match="Prompt cannot be empty."):
        cache_manager.put("", "response")
    
    # Test empty response
    with pytest.raises(ValueError, match="Response cannot be None."):
        cache_manager.put("prompt", None)
    
    # Allow empty response strings
    cache_manager.put("prompt", "")
    entry = cache_manager.get("prompt")
    assert entry.response == ""

def test_cache_special_characters(cache_manager):
    """Test handling of special characters in prompt and response."""
    special_prompt = "test\n\t\r\x00prompt🔥"
    special_response = "test\n\t\r\x00response👍"
    
    cache_manager.put(special_prompt, special_response)
    entry = cache_manager.get(special_prompt)
    
    assert entry.prompt == special_prompt
    assert entry.response == special_response

def test_cache_large_data(cache_manager):
    """Test handling of large data."""
    large_prompt = "x" * 1000000  # 1MB string
    large_response = "y" * 1000000  # 1MB string
    
    cache_manager.put(large_prompt, large_response)
    entry = cache_manager.get(large_prompt)
    
    assert entry.prompt == large_prompt
    assert entry.response == large_response

def test_database_integrity(cache_manager, temp_db_path):
    """Test database integrity and proper closing."""
    cache_manager.put("test", "response")
    
    # Force close any open connections
    del cache_manager
    
    # Try to open database directly
    with sqlite3.connect(temp_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM query_cache")
        assert cursor.fetchone()[0] == 1

def test_vacuum(cache_manager):
    """Test vacuum operation."""
    # Add and remove entries to create free space
    for i in range(100):
        cache_manager.put(f"prompt{i}", "response")
    cache_manager.clear()
    
    # Vacuum should complete without errors
    cache_manager.vacuum()