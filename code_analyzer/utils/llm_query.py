#ast-llm-code-analysis/code_analyzer/utils/llm_query.py

from typing import List, Dict, Optional, Tuple
import sqlite3
from datetime import datetime
import base64
import hashlib
import os
from pathlib import Path
from vllm import LLM, SamplingParams
import logging
from code_analyzer.utils.model_singleton import get_model

def get_two_stage_analysis(prompt: str, entity_type: str) -> tuple[str, str]:
    """
    Perform a two-stage analysis: initial analysis followed by a summary.
    
    Args:
        prompt: The initial analysis prompt
        entity_type: Type of entity being analyzed (module/class/function/method)
        
    Returns:
        Tuple of (initial analysis, summary)
    """
    # Import LLMQueryManager here to avoid circular imports
    from code_analyzer.utils.llm_query import LLMQueryManager

    # Initialize query manager with default settings
    query_manager = LLMQueryManager()
    
    # First stage: Initial analysis
    initial_analysis = query_manager.query(prompt)
    
    # Second stage: Summary
    summary_prompt = f"""
    Excellent work. Please provide a succinct summary of this {entity_type} in the following format:
    SUMMARY: 3-sentence summary goes here
    NOTES: any other comments or notes go here
    """.strip()
    
    # Create multi-turn conversation for summary
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": initial_analysis},
        {"role": "user", "content": summary_prompt}
    ]
    
    summary = query_manager.query_multi_turn(messages)
    
    return initial_analysis, summary

class LLMQueryManager:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        cache_db: str = "query_cache.db",
        debug_dir: Optional[str] = None,
        use_cache: bool = True,
        verbose: bool = False
    ):
        self.model_name = model_name
        self.cache_db = cache_db
        self.debug_dir = debug_dir
        self.use_cache = use_cache
        self.verbose = verbose
        self.model = None
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('LLMQuery')
            self.logger.setLevel(logging.INFO)
            
        # Initialize cache database when manager is created
        if self.use_cache:
            self._setup_cache_db()
      
    def _setup_cache_db(self) -> None:
        """Set up the SQLite cache database and create necessary tables."""
        if not Path(self.cache_db).parent.exists():
            Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)
            
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Add index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON query_cache(timestamp)
        """)
        
        conn.commit()
        conn.close()
        
    def _get_query_hash(self, prompt: str) -> str:
        """Generate a unique hash for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()
        
    def _get_cached_response(self, prompt: str) -> Tuple[bool, str]:
        """
        Check cache for existing response to prompt.
        
        Returns:
            Tuple of (cache_hit: bool, response: str)
        """
        if not self.use_cache:
            return False, ""
            
        query_hash = self._get_query_hash(prompt)
        
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT response FROM query_cache WHERE query_hash = ?",
            (query_hash,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return True, base64.b64decode(result[0]).decode('utf-8')
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Error decoding cached response: {e}")
                return False, ""
                
        return False, ""


from typing import List, Dict, Optional, Tuple
import logging
import sqlite3
from datetime import datetime
import base64
import hashlib
from pathlib import Path
from vllm import LLM, SamplingParams

class LLMQueryManager:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        cache_db: str = "query_cache.db",
        debug_dir: Optional[str] = None,
        llm_out_dir: Optional[str] = None,
        use_cache: bool = True,
        verbose: bool = False,
        export: bool = False
    ):
        self.model_name = model_name
        self.cache_db = cache_db
        self.debug_dir = debug_dir
        self.llm_out_dir = Path(llm_out_dir) if llm_out_dir else None
        self.use_cache = use_cache
        self.verbose = verbose
        self.export = export
        self.model = None
        
        # Configure logging
        self.logger = logging.getLogger('LLMQuery')
        if self.verbose:
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize cache database
        if self.use_cache:
            self._setup_cache_db()
            
        # Create LLM output directory if exporting
        if self.export and self.llm_out_dir:
            self.llm_out_dir.mkdir(parents=True, exist_ok=True)
            
    def _export_interaction(self, prompt: str, response: str) -> None:
        """Export LLM interaction to a text file if export is enabled."""
        if not self.export or not self.llm_out_dir:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        file_path = self.llm_out_dir / f"llm_interaction_{timestamp}.txt"
        
        separator = "="*80 + "\n" + "RESPONSE:" + "\n" + "="*80
        
        content = f"""PROMPT:
{prompt}

{separator}

{response}
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        if self.verbose:
            self.logger.info(f"Exported interaction to: {file_path}")
            
    def query(self, prompt: str, sampling_params: Optional[Dict] = None) -> str:
        """Query the LLM with verbose logging if enabled."""
        if self.verbose:
            self.logger.info("\n" + "="*80)
            self.logger.info("PROMPT:")
            self.logger.info("-"*80)
            self.logger.info(prompt)
            self.logger.info("="*80 + "\n")
        
        # Check cache first
        cache_hit, cached_response = self._get_cached_response(prompt)
        if cache_hit:
            if self.verbose:
                self.logger.info("CACHE HIT - RESPONSE:")
                self.logger.info("-"*80)
                self.logger.info(cached_response)
                self.logger.info("="*80 + "\n")
            
            # Export cache hits too if enabled
            if self.export:
                self._export_interaction(prompt, cached_response)
                
            return cached_response
        
        # Initialize model if needed
        self.initialize_model()
        
        # Set up default sampling parameters
        if sampling_params is None:
            sampling_params = {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 30000
            }
        
        # Generate response
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt}
        ]
        
        if self.verbose:
            self.logger.info("GENERATING RESPONSE...")
            
        outputs = self.model.chat(
            messages=[messages],
            sampling_params=SamplingParams(**sampling_params),
            use_tqdm=self.verbose
        )
        response = outputs[0].outputs[0].text
        
        if self.verbose:
            self.logger.info("MODEL RESPONSE:")
            self.logger.info("-"*80)
            self.logger.info(response)
            self.logger.info("="*80 + "\n")
        
        # Cache the response
        self._cache_response(prompt, response)
        
        # Export if enabled
        if self.export:
            self._export_interaction(prompt, response)
        
        return response


class LLMQueryManager:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        cache_db: str = "query_cache.db",
        debug_dir: Optional[str] = None,
        llm_out_dir: Optional[str] = None,
        use_cache: bool = True,
        verbose: bool = False,
        export: bool = False
    ):
        self.model_name = model_name
        self.cache_db = cache_db
        self.debug_dir = debug_dir
        self.llm_out_dir = Path(llm_out_dir) if llm_out_dir else None
        self.use_cache = use_cache
        self.verbose = verbose
        self.export = export
        self.model = None
        
        # Configure logging
        self.logger = logging.getLogger('LLMQuery')
        if self.verbose:
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize cache database
        if self.use_cache:
            self._setup_cache_db()
            
        # Create LLM output directory if exporting
        if self.export and self.llm_out_dir:
            self.llm_out_dir.mkdir(parents=True, exist_ok=True)
            
    def _export_interaction(self, prompt: str, response: str) -> None:
        """Export LLM interaction to a text file if export is enabled."""
        if not self.export or not self.llm_out_dir:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        file_path = self.llm_out_dir / f"llm_interaction_{timestamp}.txt"
        
        separator = "="*80 + "\n" + "RESPONSE:" + "\n" + "="*80
        
        content = f"""PROMPT:
{prompt}

{separator}

{response}
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        if self.verbose:
            self.logger.info(f"Exported interaction to: {file_path}")
       
    def _setup_cache_db(self) -> None:
        """Set up the SQLite cache database and create necessary tables."""
        if not Path(self.cache_db).parent.exists():
            Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)
            
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Add index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON query_cache(timestamp)
        """)
        
        conn.commit()
        conn.close()
        
    def _get_query_hash(self, prompt: str) -> str:
        """Generate a unique hash for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()
        
    def _get_cached_response(self, prompt: str) -> Tuple[bool, str]:
        """
        Check cache for existing response to prompt.
        
        Returns:
            Tuple of (cache_hit: bool, response: str)
        """
        if not self.use_cache:
            return False, ""
            
        query_hash = self._get_query_hash(prompt)
        
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT response FROM query_cache WHERE query_hash = ?",
            (query_hash,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return True, base64.b64decode(result[0]).decode('utf-8')
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Error decoding cached response: {e}")
                return False, ""
                
        return False, ""

    def initialize_model(self) -> None:
        """Initialize the model if not already initialized."""
        if self.model is None:
            self.model = get_model()
           
    def _cache_response(self, prompt: str, response: str) -> None:
        """Store prompt-response pair in cache."""
        if not self.use_cache:
            return
            
        query_hash = self._get_query_hash(prompt)
        timestamp = datetime.now().isoformat()
        
        # Encode as Base64 to handle special characters
        encoded_prompt = base64.b64encode(prompt.encode('utf-8')).decode('utf-8')
        encoded_response = base64.b64encode(response.encode('utf-8')).decode('utf-8')
        
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO query_cache
            (query_hash, prompt, response, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (query_hash, encoded_prompt, encoded_response, timestamp)
        )
        
        conn.commit()
        conn.close()
        
    def _write_debug_log(self, prompt: str, response: str, query_type: str = "query") -> None:
        """Write debug logs if debug directory is configured."""
        if not self.debug_dir:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        query_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        # Write prompt
        prompt_file = self.debug_dir / f"{timestamp}_{query_type}_{query_hash}_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # Write response
        response_file = self.debug_dir / f"{timestamp}_{query_type}_{query_hash}_response.txt"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(response)
 
    def query(self, prompt: str, sampling_params: Optional[Dict] = None) -> str:
        """Query the LLM with verbose logging if enabled."""
        if self.verbose:
            self.logger.info("\n" + "="*80)
            self.logger.info("PROMPT:")
            self.logger.info("-"*80)
            self.logger.info(prompt)
            self.logger.info("="*80 + "\n")
        
        # Check cache first
        cache_hit, cached_response = self._get_cached_response(prompt)
        if cache_hit:
            if self.verbose:
                self.logger.info("CACHE HIT - RESPONSE:")
                self.logger.info("-"*80)
                self.logger.info(cached_response)
                self.logger.info("="*80 + "\n")
            
            # Export cache hits too if enabled
            if self.export:
                self._export_interaction(prompt, cached_response)
                
            return cached_response
        
        # Initialize model if needed
        self.initialize_model()
        
        # Set up default sampling parameters
        if sampling_params is None:
            sampling_params = {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 30000
            }
        
        # Generate response
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt}
        ]
        
        if self.verbose:
            self.logger.info("GENERATING RESPONSE...")
            
        outputs = self.model.chat(
            messages=[messages],
            sampling_params=SamplingParams(**sampling_params),
            use_tqdm=self.verbose
        )
        response = outputs[0].outputs[0].text
        
        if self.verbose:
            self.logger.info("MODEL RESPONSE:")
            self.logger.info("-"*80)
            self.logger.info(response)
            self.logger.info("="*80 + "\n")
        
        # Cache the response
        self._cache_response(prompt, response)
        
        # Export if enabled
        if self.export:
            self._export_interaction(prompt, response)
        
        return response

    def query_multi_turn(
        self, 
        messages: List[Dict[str, str]], 
        sampling_params: Optional[Dict] = None
    ) -> str:
        """Multi-turn query with verbose logging if enabled."""
        if self.verbose:
            self.logger.info("\n" + "="*80)
            self.logger.info("MULTI-TURN CONVERSATION:")
            self.logger.info("-"*80)
            for msg in messages:
                self.logger.info(f"{msg['role'].upper()}:")
                self.logger.info(msg['content'])
                self.logger.info("-"*40)
            self.logger.info("="*80 + "\n")
        
        # Create combined prompt for caching
        combined_prompt = "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        )
        
        # Check cache first
        cache_hit, cached_response = self._get_cached_response(combined_prompt)
        if cache_hit:
            if self.verbose:
                self.logger.info("CACHE HIT - RESPONSE:")
                self.logger.info("-"*80)
                self.logger.info(cached_response)
                self.logger.info("="*80 + "\n")
            return cached_response
        
        # Initialize model if needed
        self.initialize_model()
        
        # Set up default sampling parameters
        if sampling_params is None:
            sampling_params = {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 30000
            }
        
        # Generate response
        outputs = self.model.chat(
            messages=[messages],
            sampling_params=SamplingParams(**sampling_params),
            use_tqdm=False
        )
        response = outputs[0].outputs[0].text
        
        if self.verbose:
            self.logger.info("MODEL RESPONSE:")
            self.logger.info("-"*80)
            self.logger.info(response)
            self.logger.info("="*80 + "\n")
        
        # Cache the response
        self._cache_response(combined_prompt, response)
        
        return response


