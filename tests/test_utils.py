#ast-llm-code-analysis/tests/test_utils.py

import unittest
from pathlib import Path
import sys
import os
import json
import hashlib
import base64
import sqlite3
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path to import code_analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_analyzer.utils.text_processing import (
    truncate_middle,
    truncate_above_and_below_a_string,
    write_debug_log
)
from code_analyzer.utils.llm_query import (
    setup_cache_db,
    get_query_hash,
    get_cached_response,
    cache_response,
    query_qwen,
    query_qwen_multi_turn
)

class TestTextProcessing(unittest.TestCase):
    def test_truncate_middle_short_string(self):
        """Test that strings shorter than max_length are not truncated"""
        test_str = "Short string"
        result = truncate_middle(test_str, max_length=20)
        self.assertEqual(result, test_str)
        
    def test_truncate_middle_long_string(self):
        """Test truncation of long strings"""
        test_str = "A" * 100
        max_length = 50
        result = truncate_middle(test_str, max_length=max_length)
        
        # Check length
        self.assertLessEqual(len(result), max_length)
        # Check that it contains truncation marker
        self.assertIn("TRUNCATED", result)
        # Check that it starts and ends with original content
        self.assertTrue(result.startswith("A"))
        self.assertTrue(result.endswith("A"))
        
    def test_truncate_middle_multiline(self):
        """Test truncation of multiline strings with line counting"""
        test_str = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        result = truncate_middle(test_str, max_length=20)
        self.assertIn("TRUNCATED", result)
        self.assertIn("LINES", result)
        
    def test_truncate_above_and_below_basic(self):
        """Test basic functionality of truncate_above_and_below"""
        text = "Line 1\nLine 2\nTarget\nLine 4\nLine 5"
        target = "Target"
        result = truncate_above_and_below_a_string(text, target, max_chars=30)
        
        # Target must be present
        self.assertIn(target, result)
        # Length should be less than or equal to max_chars
        self.assertLessEqual(len(result), 30)
        
    def test_truncate_above_and_below_clean_edges(self):
        """Test that clean_cut_edges parameter works correctly"""
        text = "Line 1\nLine 2\nTarget\nLine 4\nLine 5"
        target = "Target"
        result = truncate_above_and_below_a_string(text, target, max_chars=30, clean_cut_edges=True)
        
        # All lines should be complete (no partial lines)
        for line in result.split('\n'):
            self.assertIn(line, text.split('\n'))
            
    def test_truncate_above_and_below_not_found(self):
        """Test handling of target string not found"""
        text = "Line 1\nLine 2\nLine 3"
        target = "NonExistent"
        with self.assertRaises(AssertionError):
            truncate_above_and_below_a_string(text, target)
            
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_write_debug_log(self, mock_mkdir, mock_file):
        """Test debug log writing functionality"""
        prompt = "Test prompt"
        response = "Test response"
        query_type = "test_query"
        
        write_debug_log(prompt, response, query_type)
        
        # Check that directory was created
        mock_mkdir.assert_called_once_with(exist_ok=True)
        
        # Check that files were opened and written to
        self.assertEqual(mock_file.call_count, 2)  # Two files should be opened
        mock_file().write.assert_any_call(prompt)
        mock_file().write.assert_any_call(response)

class TestLLMQuery(unittest.TestCase):
    def setUp(self):
        """Set up test database"""
        self.test_db = ":memory:"
        setup_cache_db(self.test_db)
        
    def test_get_query_hash(self):
        """Test hash generation for queries"""
        test_prompt = "Test prompt"
        hash1 = get_query_hash(test_prompt)
        hash2 = get_query_hash(test_prompt)
        
        # Same input should produce same hash
        self.assertEqual(hash1, hash2)
        # Hash should be hexadecimal
        self.assertTrue(all(c in '0123456789abcdef' for c in hash1))
        
    def test_cache_and_retrieve(self):
        """Test caching and retrieving responses"""
        test_prompt = "Test prompt"
        test_response = "Test response"
        
        # Cache the response
        cache_response(test_prompt, test_response, self.test_db)
        
        # Retrieve and check
        cache_hit, retrieved = get_cached_response(test_prompt, self.test_db)
        self.assertTrue(cache_hit)
        self.assertEqual(retrieved, test_response)
        
    def test_cache_miss(self):
        """Test behavior when cache miss occurs"""
        cache_hit, retrieved = get_cached_response("Nonexistent prompt", self.test_db)
        self.assertFalse(cache_hit)
        self.assertEqual(retrieved, "")
        
    @patch('code_analyzer.utils.llm_query.LLM')
    def test_query_qwen(self, mock_llm):
        """Test basic LLM query functionality"""
        # Mock LLM response
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Test response")]
        mock_llm.return_value.chat.return_value = [mock_output]
        
        result = query_qwen("Test prompt", self.test_db)
        self.assertEqual(result, "Test response")
        
    @patch('code_analyzer.utils.llm_query.LLM')
    def test_query_qwen_multi_turn(self, mock_llm):
        """Test multi-turn conversation functionality"""
        # Mock LLM response
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Test response")]
        mock_llm.return_value.chat.return_value = [mock_output]
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = query_qwen_multi_turn(messages, self.test_db)
        self.assertEqual(result, "Test response")
        
    def test_invalid_base64_handling(self):
        """Test handling of invalid base64 in cached responses"""
        # Manually insert invalid base64
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO query_cache (query_hash, prompt, response, timestamp) VALUES (?, ?, ?, ?)",
            ("test_hash", "test_prompt", "invalid_base64!", datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        # Attempt to retrieve
        cache_hit, retrieved = get_cached_response("test_prompt", self.test_db)
        self.assertFalse(cache_hit)
        self.assertEqual(retrieved, "")

if __name__ == '__main__':
    unittest.main()