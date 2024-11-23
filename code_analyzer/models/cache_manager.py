#ast-llm-code-analysis/code_analyzer/models/cache_manager.py

from datetime import datetime
import sqlite3
import base64
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
from dataclasses import dataclass

@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    query_hash: str
    prompt: str
    response: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

    @classmethod
    def from_db_row(cls, row: tuple) -> 'CacheEntry':
        """Create a CacheEntry from a database row."""
        query_hash, prompt, response, timestamp, metadata = row
        return cls(
            query_hash=query_hash,
            prompt=base64.b64decode(prompt).decode('utf-8'),
            response=base64.b64decode(response).decode('utf-8'),
            timestamp=datetime.fromisoformat(timestamp),
            metadata=json.loads(metadata) if metadata else None
        )

class CacheManager:
    """Manages SQLite-based caching for LLM queries and responses."""
    
    def __init__(self, db_path: str = "query_cache.db"):
        """
        Initialize the cache manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._setup_database()

    def _setup_database(self) -> None:
        """Set up the SQLite database and required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            # Add indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON query_cache(timestamp)
            """)
            conn.commit()

    @staticmethod
    def _generate_hash(prompt: str) -> str:
        """
        Generate a unique hash for a given prompt.
        
        Args:
            prompt: The input prompt to hash
            
        Returns:
            SHA-256 hash of the prompt
        """
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> Optional[CacheEntry]:
        """
        Retrieve a cached response for the given prompt.
        
        Args:
            prompt: The input prompt to look up
            
        Returns:
            CacheEntry if found, None otherwise
        """
        query_hash = self._generate_hash(prompt)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT query_hash, prompt, response, timestamp, metadata 
                FROM query_cache 
                WHERE query_hash = ?
                """,
                (query_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                return CacheEntry.from_db_row(row)
            return None

    def put(self, prompt: str, response: str, metadata: Dict[str, Any] = None) -> None:
        """
        Store a prompt-response pair in the cache.
        
        Args:
            prompt: The input prompt
            response: The generated response
            metadata: Optional metadata to store with the cache entry
        """
        query_hash = self._generate_hash(prompt)
        timestamp = datetime.now().isoformat()
        
        # Encode prompt and response as Base64
        encoded_prompt = base64.b64encode(prompt.encode('utf-8')).decode('utf-8')
        encoded_response = base64.b64encode(response.encode('utf-8')).decode('utf-8')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO query_cache 
                (query_hash, prompt, response, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query_hash,
                    encoded_prompt,
                    encoded_response,
                    timestamp,
                    json.dumps(metadata) if metadata else None
                )
            )
            conn.commit()

    def invalidate(self, prompt: str) -> bool:
        """
        Remove a specific entry from the cache.
        
        Args:
            prompt: The prompt whose cache entry should be invalidated
            
        Returns:
            True if an entry was removed, False otherwise
        """
        query_hash = self._generate_hash(prompt)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM query_cache WHERE query_hash = ?",
                (query_hash,)
            )
            return cursor.rowcount > 0

    def clear(self, older_than: Optional[datetime] = None) -> int:
        """
        Clear the cache, optionally only entries older than a specified time.
        
        Args:
            older_than: Optional datetime, if provided only entries older than
                       this will be cleared
                       
        Returns:
            Number of entries cleared
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if older_than:
                cursor.execute(
                    "DELETE FROM query_cache WHERE timestamp < ?",
                    (older_than.isoformat(),)
                )
            else:
                cursor.execute("DELETE FROM query_cache")
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            stats = {
                'total_entries': cursor.execute(
                    "SELECT COUNT(*) FROM query_cache"
                ).fetchone()[0],
                'oldest_entry': cursor.execute(
                    "SELECT MIN(timestamp) FROM query_cache"
                ).fetchone()[0],
                'newest_entry': cursor.execute(
                    "SELECT MAX(timestamp) FROM query_cache"
                ).fetchone()[0],
                'total_size_bytes': cursor.execute(
                    "SELECT SUM(LENGTH(prompt) + LENGTH(response)) FROM query_cache"
                ).fetchone()[0] or 0
            }
            return stats

    def vacuum(self) -> None:
        """Optimize the database by running VACUUM."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")

    def __enter__(self) -> 'CacheManager':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.vacuum()

# Example usage:
if __name__ == "__main__":
    # Use as a context manager
    with CacheManager() as cache:
        # Store a response
        cache.put("What is 2+2?", "4", metadata={"type": "math", "confidence": 1.0})
        
        # Retrieve from cache
        if entry := cache.get("What is 2+2?"):
            print(f"Cached response: {entry.response}")
            print(f"Timestamp: {entry.timestamp}")
            print(f"Metadata: {entry.metadata}")
        
        # Get cache statistics
        stats = cache.get_stats()
        print(f"Cache statistics: {stats}")