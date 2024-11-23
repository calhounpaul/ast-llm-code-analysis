#ast-llm-code-analysis/tests/test_cache_manager.py

import pytest
import os
import sqlite3
import base64
from datetime import datetime
from pathlib import Path
from code_analyzer.models.cache_manager import (
    setup_cache_db,
    get_query_hash,
    get_cached_response,
    cache_response
)

# Test fixtures
@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture to create a temporary database path"""
    return str(tmp_path / "test_cache.db")

@pytest.fixture
def clean_db(temp_db_path):
    """Fixture to ensure a clean database for each test"""
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)
    setup_cache_db(temp_db_path)
    return temp_db_path

def test_setup_cache_db(temp_db_path):
    """Test database setup creates the correct table structure"""
    # Setup database
    setup_cache_db(temp_db_path)
    
    # Verify table exists with correct schema
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("PRAGMA table_info(query_cache)")
    columns = cursor.fetchall()
    
    # Verify expected columns
    column_names = [col[1] for col in columns]
    expected_columns = ['query_hash', 'prompt', 'response', 'timestamp']
    assert all(col in column_names for col in expected_columns)
    
    # Verify primary key
    primary_key_cols = [col[1] for col in columns if col[5]]  # col[5] is pk flag
    assert primary_key_cols == ['query_hash']
    
    conn.close()

def test_get_query_hash():
    """Test hash generation is consistent and correct"""
    test_prompt = "Test prompt content"
    
    # Hash should be consistent
    hash1 = get_query_hash(test_prompt)
    hash2 = get_query_hash(test_prompt)
    assert hash1 == hash2
    
    # Different prompts should have different hashes
    different_prompt = "Different content"
    different_hash = get_query_hash(different_prompt)
    assert hash1 != different_hash
    
    # Hash should be a valid SHA-256 hash (64 characters, hex)
    assert len(hash1) == 64
    assert all(c in '0123456789abcdef' for c in hash1)

def test_cache_response(clean_db):
    """Test storing responses in cache"""
    # Test data
    prompt = "Test prompt"
    response = "Test response"
    
    # Cache the response
    cache_response(prompt, response, clean_db)
    
    # Verify data was stored correctly
    conn = sqlite3.connect(clean_db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT prompt, response FROM query_cache WHERE query_hash=?",
                  (get_query_hash(prompt),))
    result = cursor.fetchone()
    
    # Decode stored data
    stored_prompt = base64.b64decode(result[0]).decode('utf-8')
    stored_response = base64.b64decode(result[1]).decode('utf-8')
    
    assert stored_prompt == prompt
    assert stored_response == response
    
    conn.close()

def test_get_cached_response(clean_db):
    """Test retrieving responses from cache"""
    # Test data
    prompt = "Test prompt"
    response = "Test response"
    
    # First verify no response exists
    found, _ = get_cached_response(prompt, clean_db)
    assert not found
    
    # Cache a response
    cache_response(prompt, response, clean_db)
    
    # Verify we can retrieve it
    found, retrieved_response = get_cached_response(prompt, clean_db)
    assert found
    assert retrieved_response == response

def test_cache_update(clean_db):
    """Test updating existing cache entries"""
    prompt = "Test prompt"
    response1 = "First response"
    response2 = "Updated response"
    
    # Cache initial response
    cache_response(prompt, response1, clean_db)
    
    # Cache updated response
    cache_response(prompt, response2, clean_db)
    
    # Verify only updated response exists
    conn = sqlite3.connect(clean_db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM query_cache WHERE query_hash=?",
                  (get_query_hash(prompt),))
    count = cursor.fetchone()[0]
    assert count == 1  # Should only have one entry
    
    # Verify it's the updated response
    found, retrieved_response = get_cached_response(prompt, clean_db)
    assert found
    assert retrieved_response == response2
    
    conn.close()

def test_cache_with_special_characters(clean_db):
    """Test caching with special characters and unicode"""
    prompt = "Test prompt 🚀 with special chars: é è ñ"
    response = "Response with unicode: 你好 and symbols: ®™"
    
    cache_response(prompt, response, clean_db)
    found, retrieved_response = get_cached_response(prompt, clean_db)
    
    assert found
    assert retrieved_response == response

def test_invalid_db_path():
    """Test handling of invalid database paths"""
    invalid_path = "/nonexistent/path/db.sqlite"
    
    # Should raise an exception when trying to setup
    with pytest.raises(sqlite3.OperationalError):
        setup_cache_db(invalid_path)

def test_timestamp_recording(clean_db):
    """Test that timestamps are properly recorded"""
    prompt = "Test prompt"
    response = "Test response"
    
    # Cache response
    cache_response(prompt, response, clean_db)
    
    # Verify timestamp exists and is recent
    conn = sqlite3.connect(clean_db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT timestamp FROM query_cache WHERE query_hash=?",
                  (get_query_hash(prompt),))
    timestamp_str = cursor.fetchone()[0]
    
    # Parse timestamp
    timestamp = datetime.fromisoformat(timestamp_str)
    now = datetime.now()
    
    # Timestamp should be recent (within last minute)
    time_diff = now - timestamp
    assert time_diff.total_seconds() < 60
    
    conn.close()