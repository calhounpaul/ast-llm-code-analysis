#!/usr/bin/env python3
import os
from pathlib import Path
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
import json
import ast
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
import sys, torch
from collections import defaultdict
import textwrap
import sqlite3
from datetime import datetime
import base64
import hashlib
from vllm import LLM, SamplingParams
from typing import List, Dict
import os

THIS_DIR = Path(__file__).resolve().parent

model_name = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"

# Add these new templates at the top with the other PROMPT_TEMPLATES
SUMMARY_PROMPT = """
Excellent work. Please provide a succinct summary of this {entity_type} in the following format:
SUMMARY: 3-sentence summary goes here
NOTES: any other comments or notes go here
""".strip()

PROMPT_TEMPLATES = {
    'function': textwrap.dedent("""
        {human_readable_tree}

        From the file `/base/{path}`, analyze this Python function in context:
        
        {raw_code}
        
        Function Context:
        - Defined in: {defined_in}
        - Calls: {calls}
        - Uses variables: {reads}
        - Modifies variables: {writes}
        - Raises exceptions: {raises}
        - Catches exceptions: {catches}
        - Type hints: {type_hints}
        - File overview: {node_map}
        
        1. What is the main purpose of this function?
        2. How does it interact with its dependencies and environment?
        3. What are the key steps in its implementation?
        4. What are its inputs, outputs, and side effects?
        5. What are the notable edge cases, error conditions, or error handling?
        
        Please provide a concise analysis that covers these aspects.
    """).strip(),
    
    'method': textwrap.dedent("""
        {human_readable_tree}

        From the file `/base/{path}`, analyze this Python class method in context:
        
        {raw_code}
        
        Method Context:
        - Defined in class: {defined_in}
        - Calls: {calls}
        - Uses class/instance variables: {reads}
        - Modifies class/instance variables: {writes}
        - Raises exceptions: {raises}
        - Catches exceptions: {catches}
        - Type hints: {type_hints}
        - File overview: {node_map}
        
        1. What is the main purpose of this method?
        2. How does it interact with the class's state and other methods?
        3. What other methods or class attributes does it use?
        4. What are its inputs, outputs, and side effects on class state?
        5. What are the notable edge cases, error conditions, or error handling?
        
        Please provide a concise analysis that covers these aspects.
    """).strip(),
    
    'class': textwrap.dedent("""
        {human_readable_tree}

        From the file `/base/{path}`, analyze this Python class in context:
        
        {raw_code}
        
        Class Context:
        - Defined in: {defined_in}
        - Inherits from: {inherits}
        - Uses decorators: {decorators}
        - Contains methods: {contains}
        - Imports: {imports}
        - File overview: {node_map}
        
        1. What is the main purpose and responsibility of this class?
        2. How does it use inheritance and what patterns does it implement?
        3. What are its key attributes, methods, and their relationships?
        4. How does it manage and protect its internal state?
        5. What are its key dependencies and interactions with other components?
        
        Please provide a concise analysis that covers these aspects.
    """).strip(),
    
    'module': textwrap.dedent("""
        {human_readable_tree}

        From the file `/base/{path}`, analyze this Python module in context:
        
        {raw_code}
        
        Module Context:
        - Direct imports: {direct_imports}
        - From imports: {from_imports}
        - Defines classes: {classes}
        - Defines functions: {functions}
        - Defines constants: {constants}
        - File overview: {node_map}
        
        1. What is the main purpose and responsibility of this module?
        2. How do its components work together to achieve its purpose?
        3. What are its key dependencies and how does it use them?
        4. What patterns or principles does it implement?
        5. How does it fit into the larger codebase architecture?
        
        Please provide a concise analysis that covers these aspects.
    """).strip()
}

def setup_cache_db(db_path: str = "query_cache.db") -> None:
    """
    Set up the SQLite database for caching queries and responses.
    Creates the database and table if they don't exist.
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def get_query_hash(prompt: str) -> str:
    """
    Generate a unique hash for a given prompt.
    
    Args:
        prompt: The input prompt to hash
        
    Returns:
        SHA-256 hash of the prompt
    """
    return hashlib.sha256(prompt.encode()).hexdigest()

def get_cached_response(prompt: str, db_path: str = "query_cache.db") -> tuple[bool, str]:
    """
    Check if a response for the given prompt exists in the cache.
    Returns Base64 decoded response if found.
    
    Args:
        prompt: The input prompt to look up
        db_path: Path to the SQLite database file
        
    Returns:
        Tuple of (bool, str) indicating if cache hit and the decoded response
    """
    query_hash = get_query_hash(prompt)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT response FROM query_cache WHERE query_hash = ?",
        (query_hash,)
    )
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        # Decode the Base64 response
        try:
            decoded_response = base64.b64decode(result[0]).decode('utf-8')
            return True, decoded_response
        except Exception as e:
            print(f"Error decoding response: {e}")
            return False, ""
    return False, ""

def cache_response(prompt: str, response: str, db_path: str = "query_cache.db") -> None:
    """
    Store a prompt-response pair in the cache with Base64 encoding.
    
    Args:
        prompt: The input prompt
        response: The generated response
        db_path: Path to the SQLite database file
    """
    query_hash = get_query_hash(prompt)
    timestamp = datetime.now().isoformat()
    
    # Encode prompt and response as Base64
    encoded_prompt = base64.b64encode(prompt.encode('utf-8')).decode('utf-8')
    encoded_response = base64.b64encode(response.encode('utf-8')).decode('utf-8')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        INSERT OR REPLACE INTO query_cache (query_hash, prompt, response, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (query_hash, encoded_prompt, encoded_response, timestamp)
    )
    
    conn.commit()
    conn.close()

# Add at the top of the file, after the imports:
DEBUG_TO_TXT = True  # Set to False to disable debug logging

def write_debug_log(prompt: str, response: str, query_type: str = "query") -> None:
    """
    Write debug logs for LLM queries and responses to text files.
    
    Args:
        prompt: The input prompt
        response: The generated response
        query_type: Type of query for filename prefix (default: "query")
    """
    if not DEBUG_TO_TXT:
        return
        
    debug_dir = THIS_DIR / 'debug'
    debug_dir.mkdir(exist_ok=True)
    
    # Create a timestamp and sanitized hash for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    query_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    # Write prompt
    prompt_file = debug_dir / f"{timestamp}_{query_type}_{query_hash}_prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    # Write response
    response_file = debug_dir / f"{timestamp}_{query_type}_{query_hash}_response.txt"
    with open(response_file, 'w', encoding='utf-8') as f:
        f.write(response)

def query_qwen(prompt: str, db_path: str = "query_cache.db", use_cache: bool = True) -> str:
    """
    Query the Qwen model with caching support using VLLM.
    Args:
        prompt: The input prompt
        db_path: Path to the SQLite database file
        use_cache: Whether to use caching (default: True)
    Returns:
        Generated response from model or cache
    """
    # Ensure cache database exists
    if use_cache and not os.path.exists(db_path):
        setup_cache_db(db_path)

    # Check cache first if enabled
    if use_cache:
        print("Checking cache...")
        cache_hit, cached_response = get_cached_response(prompt, db_path)
        if cache_hit:
            if DEBUG_TO_TXT:
                write_debug_log(prompt, cached_response, "cache_hit")
            return cached_response

    # Initialize VLLM model and sampling parameters
    #llm = LLM(model=model_name,trust_remote_code=True)  # Adjust model path as needed
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=30000
    )

    # Create messages for chat format
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt}
    ]
    global model
    # Generate response using VLLM
    outputs = model.chat(
        messages=[messages],
        sampling_params=sampling_params,
        use_tqdm=False
    )
    response = outputs[0].outputs[0].text

    # Write debug logs before caching
    if DEBUG_TO_TXT:
        write_debug_log(prompt, response, "model_query")

    # Cache the response if caching is enabled
    if use_cache:
        cache_response(prompt, response, db_path)

    return response

def query_qwen_multi_turn(messages: List[Dict[str, str]], db_path: str = "query_cache.db", use_cache: bool = True) -> str:
    """
    Query the Qwen model with support for multi-turn conversations and caching using VLLM.
    Args:
        messages: List of message dictionaries with role and content
        db_path: Path to the SQLite database file
        use_cache: Whether to use caching (default: True)
    Returns:
        Generated response from model or cache
    """
    # Create a combined prompt for caching purposes
    combined_prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

    # Ensure cache database exists
    if use_cache and not os.path.exists(db_path):
        setup_cache_db(db_path)

    # Check cache first if enabled
    if use_cache:
        print("Checking cache...")
        cache_hit, cached_response = get_cached_response(combined_prompt, db_path)
        if cache_hit:
            if DEBUG_TO_TXT:
                write_debug_log(combined_prompt, cached_response, "cache_hit_multi")
            return cached_response

    # Initialize VLLM model and sampling parameters
    #llm = LLM(model="Qwen/Qwen-7B")  # Adjust model path as needed
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=30000
    )
    global model
    # Generate response using VLLM
    outputs = model.chat(
        messages=[messages],  # Wrap in list since VLLM expects batch input
        sampling_params=sampling_params,
        use_tqdm=False
    )
    response = outputs[0].outputs[0].text

    # Write debug logs before caching
    if DEBUG_TO_TXT:
        write_debug_log(combined_prompt, response, "model_query_multi")

    # Cache the response if caching is enabled
    if use_cache:
        cache_response(combined_prompt, response, db_path)

    return response

def get_two_stage_analysis(prompt: str, entity_type: str) -> tuple[str, str]:
    """
    Perform a two-stage analysis: initial analysis followed by a summary.
    
    Args:
        prompt: The initial analysis prompt
        entity_type: Type of entity being analyzed (module/class/function/method)
        
    Returns:
        Tuple of (initial analysis, summary)
    """
    # First stage: Initial analysis
    initial_analysis = query_qwen(prompt)
    
    # Second stage: Summary
    summary_prompt = SUMMARY_PROMPT.format(entity_type=entity_type)
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": initial_analysis},
        {"role": "user", "content": summary_prompt}
    ]
    summary = query_qwen_multi_turn(messages)
    
    return initial_analysis, summary

@dataclass
class AnalysisResults:
    initial_analysis: Optional[str] = None
    summary: Optional[str] = None

# Update the dataclasses (add this field to ModuleInfo, ClassInfo, and FunctionInfo)
@dataclass
class FunctionInfo:
    name: str
    parameters: List['Parameter']
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_method: bool = False
    is_static: bool = False
    is_class_method: bool = False
    is_property: bool = False
    calls: List[str] = field(default_factory=list)
    assigns_to: List[str] = field(default_factory=list)
    reads_from: List[str] = field(default_factory=list)
    raw_code: str = ""
    prompt: str = ""
    llm_analysis: Optional[AnalysisResults] = None
    
    def to_compact(self) -> Dict[str, Any]:
        output = CompactOutput()
        output.add('name', self.name)
        output.add('parameters', [p.to_compact() for p in self.parameters])
        output.add('return_annotation', self.return_annotation)
        output.add('decorators', self.decorators)
        output.add('docstring', self.docstring)
        output.add('raw_code', self.raw_code)
        output.add('prompt', self.prompt)
        if self.llm_analysis:
            output.add('llm_analysis', {
                'initial_analysis': self.llm_analysis.initial_analysis,
                'summary': self.llm_analysis.summary
            })
        
        # Only include boolean flags that are True
        for flag in ['is_method', 'is_static', 'is_class_method', 'is_property']:
            if getattr(self, flag):
                output.add(flag, True)
                
        output.add('calls', self.calls)
        output.add('assigns_to', self.assigns_to)
        output.add('reads_from', self.reads_from)
        return output.to_dict()

@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    methods: Dict[str, 'FunctionInfo'] = field(default_factory=dict)
    properties: List[str] = field(default_factory=list)
    class_variables: Dict[str, str] = field(default_factory=dict)
    instance_variables: Set[str] = field(default_factory=set)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    raw_code: str = ""
    prompt: str = ""
    llm_analysis: Optional[AnalysisResults] = None
    
    def to_compact(self) -> Dict[str, Any]:
        output = CompactOutput()
        output.add('name', self.name)
        output.add('bases', self.bases)
        output.add('raw_code', self.raw_code)
        output.add('prompt', self.prompt)
        if self.llm_analysis:
            output.add('llm_analysis', {
                'initial_analysis': self.llm_analysis.initial_analysis,
                'summary': self.llm_analysis.summary
            })
        
        if self.methods:
            output.add('methods', {
                name: method.to_compact() 
                for name, method in self.methods.items()
            })
        
        output.add('properties', self.properties)
        output.add('class_variables', self.class_variables)
        output.add('instance_variables', sorted(list(self.instance_variables)))
        output.add('decorators', self.decorators)
        output.add('docstring', self.docstring)
        return output.to_dict()

@dataclass
class ModuleInfo:
    path: str
    imports: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    classes: Dict[str, 'ClassInfo'] = field(default_factory=dict)
    functions: Dict[str, 'FunctionInfo'] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    globals: Dict[str, Any] = field(default_factory=dict)
    class_instantiations: List[Dict[str, Union[str, List[str]]]] = field(default_factory=list)
    raw_code: str = ""
    prompt: str = ""
    error: Optional[str] = None
    llm_analysis: Optional[AnalysisResults] = None
    
    def to_compact(self) -> Dict[str, Any]:
        output = CompactOutput()
        output.add('path', self.path)
        output.add('raw_code', self.raw_code)
        output.add('prompt', self.prompt)
        if self.llm_analysis:
            output.add('llm_analysis', {
                'initial_analysis': self.llm_analysis.initial_analysis,
                'summary': self.llm_analysis.summary
            })
        
        # Only include non-empty import types
        imports = {k: v for k, v in self.imports.items() if v}
        if imports:
            output.add('imports', imports)
            
        if self.classes:
            output.add('classes', {
                name: cls.to_compact() 
                for name, cls in self.classes.items()
            })
            
        if self.functions:
            output.add('functions', {
                name: func.to_compact() 
                for name, func in self.functions.items()
            })
            
        output.add('constants', self.constants)
        output.add('globals', self.globals)
        output.add('class_instantiations', self.class_instantiations)
        return output.to_dict()

# Modify the to_compact methods to handle the new structure
@dataclass
class Parameter:
    name: str
    annotation: Optional[str] = None
    default_value: Optional[str] = None
    
    def to_compact(self) -> Dict[str, Any]:
        output = CompactOutput()
        output.add('name', self.name)
        output.add('annotation', self.annotation)
        output.add('default_value', self.default_value)
        return output.to_dict()

@dataclass
class CompactOutput:
    """Helper class to manage compact output format"""
    data: Dict[str, Any] = field(default_factory=dict)
    empty_keys: Set[str] = field(default_factory=set)
    
    def add(self, key: str, value: Any) -> None:
        if value:
            if isinstance(value, dict) and not any(value.values()):
                self.empty_keys.add(key)
            elif isinstance(value, (list, set)) and not value:
                self.empty_keys.add(key)
            else:
                self.data[key] = value
                
    def to_dict(self) -> Dict[str, Any]:
        if self.empty_keys:
            self.data['empty_keys'] = sorted(list(self.empty_keys))
        return self.data

def truncate_middle(s: str, max_length: int = 4000) -> str:
    """
    Truncate a string in the middle to a maximum length.
    
    Args:
        s: The input string to truncate
        max_length: The maximum length of the output string
    
    Returns:
        Truncated string with placeholder in the middle
    """
    if len(s) <= max_length:
        return s
    excess_chars = len(s) - max_length
    start = (max_length // 2) - (excess_chars // 2)
    end = start + excess_chars
    middle_str = s[start:end]
    excess_lines = middle_str.count('\n')
    return s[:start] + f" ... TRUNCATED {excess_lines} LINES AND {excess_chars} CHARACTERS ... " + s[end:]

def extract_human_readable_file_tree(repo_path: Path) -> str:
    """
    Extract a human-readable file tree structure for a given folder
    """
    tree = []
    for path in repo_path.rglob('*'):
        if path.is_file():
            tree.append("/base/"+str(path.relative_to(repo_path)))
    return "\n".join(sorted(tree))

def truncate_above_and_below_a_string(s: str, target: str, max_chars: int = 600, clean_cut_edges: bool = True) -> str:
    """
    Truncate a string at two locations while preserving a target line and its context:
    1) the region between the first line and the target line
    2) the region between the target line and the last line
    
    The function attempts to keep a balanced amount of content above and below the target,
    proportional to their original sizes.
    
    Args:
        s: The input string to truncate
        target: The target line to keep
        max_chars: Maximum number of characters to keep in the final string
        clean_cut_edges: Whether to ensure truncation occurs at line boundaries
    
    Returns:
        Truncated string containing the target line and surrounding context
        
    Raises:
        AssertionError: If target string is not found in input string
    """
    total_chars = len(s)
    if total_chars <= max_chars:
        return s
        
    index_of_first_char_of_target = s.find(target)
    assert index_of_first_char_of_target != -1, f"target not found in string: {target}"
    
    # Calculate balanced truncation points
    chars_before_target = index_of_first_char_of_target
    chars_after_target = total_chars - (index_of_first_char_of_target + len(target))
    
    # Calculate how much to remove from each section
    qty_to_remove_total = total_chars - max_chars
    if qty_to_remove_total <= 0:
        return s
        
    fraction_before = chars_before_target / total_chars
    qty_to_remove_before = round(fraction_before * qty_to_remove_total)
    qty_to_remove_after = qty_to_remove_total - qty_to_remove_before
    
    # Calculate cut points for before target
    if chars_before_target > 0:
        halfway_before = chars_before_target // 2
        start_upper_cut = max(0, halfway_before - qty_to_remove_before // 2)
        end_upper_cut = min(index_of_first_char_of_target, 
                          halfway_before + qty_to_remove_before // 2)
    else:
        start_upper_cut = end_upper_cut = 0
    
    # Calculate cut points for after target
    target_end_index = index_of_first_char_of_target + len(target)
    if chars_after_target > 0:
        halfway_after = target_end_index + (chars_after_target // 2)
        start_lower_cut = max(target_end_index, 
                            halfway_after - qty_to_remove_after // 2)
        end_lower_cut = min(total_chars, 
                          halfway_after + qty_to_remove_after // 2)
    else:
        start_lower_cut = end_lower_cut = total_chars
    
    # Extract chunks
    first_chunk = s[:start_upper_cut]
    middle_chunk = s[end_upper_cut:start_lower_cut]
    last_chunk = s[end_lower_cut:]
    
    # Clean cut at line boundaries if requested
    if clean_cut_edges:
        if first_chunk:
            last_newline = first_chunk.rfind('\n')
            if last_newline != -1:
                first_chunk = first_chunk[:last_newline + 1]
            else:
                first_chunk = ""
                
        if middle_chunk:
            first_newline = middle_chunk.find('\n')
            last_newline = middle_chunk.rfind('\n')
            if first_newline != -1 and last_newline != -1:
                middle_chunk = middle_chunk[first_newline:last_newline + 1]
                
        if last_chunk:
            first_newline = last_chunk.find('\n')
            if first_newline != -1:
                last_chunk = last_chunk[first_newline:]
            else:
                last_chunk = ""
    
    # Build final string
    parts = []
    if first_chunk:
        parts.extend([first_chunk, "... TRUNCATED ..."])
    parts.append(middle_chunk)
    if last_chunk:
        parts.extend(["... TRUNCATED ...", last_chunk])
    
    return "".join(parts).strip()

def extract_map_of_nodes(source: str, max_depth: int = 3) -> Dict[str, Any]:
    """Extract a map of functions, classes, imports, and other nodes from source code"""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        if "f-string expression part cannot include a backslash" in str(e):
            # Remove backslashes from f-strings and try again
            source = source.replace("\\", "")
            tree = ast.parse(source)
        else:
            raise e
    node_map = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_map[node.__class__.__name__].append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            node_map[node.__class__.__name__].extend(alias.name for alias in node.names)
        if len(node_map) >= max_depth:
            break
    return dict(node_map)

@dataclass
class NodeRelationships:
    """Tracks relationships between AST nodes"""
    # Direct dependencies
    calls: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    inherits_from: Set[str] = field(default_factory=set)
    
    # Variable relationships
    reads_from: Set[str] = field(default_factory=set)
    writes_to: Set[str] = field(default_factory=set)
    
    # Structural relationships
    defined_in: Optional[str] = None
    contains: Set[str] = field(default_factory=set)
    
    # Type information
    type_hints: Dict[str, str] = field(default_factory=dict)
    
    # Control flow
    raises: Set[str] = field(default_factory=set)
    catches: Set[str] = field(default_factory=set)
    
    def merge(self, other: 'NodeRelationships') -> None:
        """Merge another NodeRelationships object into this one"""
        self.calls.update(other.calls)
        self.imports.update(other.imports)
        self.inherits_from.update(other.inherits_from)
        self.reads_from.update(other.reads_from)
        self.writes_to.update(other.writes_to)
        self.contains.update(other.contains)
        self.raises.update(other.raises)
        self.catches.update(other.catches)
        self.type_hints.update(other.type_hints)

class RelationshipVisitor(ast.NodeVisitor):
    def __init__(self):
        self.relationships: Dict[str, NodeRelationships] = defaultdict(NodeRelationships)
        self.current_scope: List[str] = []
        self.scope_stack: List[str] = []
    
    @property
    def current_scope_name(self) -> str:
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class relationships including inheritance and decorators"""
        class_name = node.name
        relationships = self.relationships[class_name]
        
        # Track inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                relationships.inherits_from.add(base.id)
            elif isinstance(base, ast.Attribute):
                relationships.inherits_from.add(ast.unparse(base))
        
        # Track decorator relationships
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                relationships.calls.add(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    relationships.calls.add(decorator.func.id)
        
        # Track scope
        self.scope_stack.append(class_name)
        relationships.defined_in = self.current_scope_name
        
        # Visit class body
        self.generic_visit(node)
        
        self.scope_stack.pop()
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function relationships including calls, variables, and types"""
        func_name = node.name
        relationships = self.relationships[func_name]
        
        # Track scope
        self.scope_stack.append(func_name)
        relationships.defined_in = self.current_scope_name
        
        # Track return type annotation
        if node.returns:
            relationships.type_hints['return'] = ast.unparse(node.returns)
        
        # Track parameter type hints
        for arg in node.args.args:
            if arg.annotation:
                relationships.type_hints[arg.arg] = ast.unparse(arg.annotation)
        
        # Visit function body
        self.generic_visit(node)
        
        self.scope_stack.pop()
    
    def visit_Call(self, node: ast.Call) -> None:
        """Track function and method calls"""
        if self.scope_stack:
            caller = self.relationships[self.scope_stack[-1]]
            if isinstance(node.func, ast.Name):
                caller.calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                caller.calls.add(ast.unparse(node.func))
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        """Track variable reads and writes"""
        if self.scope_stack:
            current = self.relationships[self.scope_stack[-1]]
            if isinstance(node.ctx, ast.Load):
                current.reads_from.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                current.writes_to.add(node.id)
        self.generic_visit(node)
    
    def visit_Raise(self, node: ast.Raise) -> None:
        """Track raised exceptions"""
        if self.scope_stack and node.exc:
            current = self.relationships[self.scope_stack[-1]]
            if isinstance(node.exc, ast.Name):
                current.raises.add(node.exc.id)
            elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                current.raises.add(node.exc.func.id)
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Track caught exceptions"""
        if self.scope_stack and node.type:
            current = self.relationships[self.scope_stack[-1]]
            if isinstance(node.type, ast.Name):
                current.catches.add(node.type.id)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-imports"""
        if self.scope_stack:
            current = self.relationships[self.scope_stack[-1]]
            module = node.module or ''
            for alias in node.names:
                current.imports.add(f"{module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Track direct imports"""
        if self.scope_stack:
            current = self.relationships[self.scope_stack[-1]]
            for alias in node.names:
                current.imports.add(alias.name)
        self.generic_visit(node)

def analyze_relationships(source_code: str) -> Dict[str, NodeRelationships]:
    """
    Analyze relationships between nodes in Python source code.
    
    Args:
        source_code: Python source code to analyze
        
    Returns:
        Dictionary mapping node names to their relationships
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        if "f-string expression part cannot include a backslash" in str(e):
            # Remove backslashes from f-strings and try again
            source_code = source_code.replace("\\", "")
            tree = ast.parse(source_code)
        else:
            raise e
    visitor = RelationshipVisitor()
    visitor.visit(tree)
    return visitor.relationships

class ASTAnalyzer(ast.NodeVisitor):
    def __init__(self, source: str, file_path: str, relationships: Dict[str, NodeRelationships], repo_path: Path, human_readable_tree: str):
        self.source = source
        self.repo_path = repo_path
        self.file_path = file_path
        self.relationships = relationships
        self.current_class: Optional[ClassInfo] = None
        self.current_function: Optional[FunctionInfo] = None
        self.module = ModuleInfo(path=file_path, raw_code=source)
        self.line_offset = 0
        self.node_map = truncate_middle(str(extract_map_of_nodes(source)), max_length=600)
        self.human_readable_tree = human_readable_tree
    
    def _format_list(self, items: Union[List, Set, None]) -> str:
        """Format a list of items for prompt display"""
        if not items:
            return "None"
        return ", ".join(sorted(items))
    
    def _format_dict(self, d: Dict) -> str:
        """Format a dictionary for prompt display"""
        if not d:
            return "None"
        return ", ".join(f"{k}: {v}" for k, v in sorted(d.items()))
    
    def get_node_source(self, node: ast.AST) -> str:
        """Extract the source code for a given node"""
        lines = self.source.splitlines()
        return '\n'.join(lines[node.lineno - 1:node.end_lineno])
    
    def visit_Module(self, node: ast.Module) -> None:
        if ast.get_docstring(node):
            self.module.docstring = ast.get_docstring(node)

        module_context = {
            'path': Path(self.file_path).relative_to(self.repo_path),
            'raw_code': truncate_middle(self.module.raw_code),
            'direct_imports': self._format_list(self.module.imports.get('direct', [])),
            'from_imports': self._format_list(self.module.imports.get('from', [])),
            'classes': self._format_list(self.module.classes.keys()),
            'functions': self._format_list(self.module.functions.keys()),
            'constants': self._format_list(self.module.constants.keys()),
            'globals': self._format_list(self.module.globals.keys()),
            'class_instantiations': self._format_list(self.module.class_instantiations),
            'node_map': self.node_map,
            'human_readable_tree': truncate_above_and_below_a_string(self.human_readable_tree, str(Path(self.file_path).relative_to(self.repo_path))),
        }

        self.module.prompt = PROMPT_TEMPLATES['module'].format(**module_context)
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.module.imports['direct'].append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ''
        for alias in node.names:
            self.module.imports['from'].append(f"{module}.{alias.name}")

    def _get_annotation_str(self, annotation: Optional[ast.AST]) -> Optional[str]:
        if annotation is None:
            return None
        return ast.unparse(annotation)

    def _get_value_str(self, value: Optional[ast.AST]) -> Optional[str]:
        if value is None:
            return None
        return ast.unparse(value)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raw_code = self.get_node_source(node)
        class_rels = self.relationships[node.name]
        
        # Format class context properly
        class_context = {
            'path': Path(self.file_path).relative_to(self.repo_path),
            'raw_code': raw_code,
            'defined_in': class_rels.defined_in or "None",
            'inherits': self._format_list(class_rels.inherits_from),
            'decorators': self._format_list(class_rels.calls),
            'contains': self._format_list(class_rels.contains),
            'imports': self._format_list(class_rels.imports),
            'node_map': self.node_map,
            'human_readable_tree': truncate_above_and_below_a_string(self.human_readable_tree, str(Path(self.file_path).relative_to(self.repo_path))),
        }
        
        class_info = ClassInfo(
            name=node.name,
            bases=[ast.unparse(base) for base in node.bases],
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            docstring=ast.get_docstring(node),
            raw_code=raw_code,
            prompt=PROMPT_TEMPLATES['class'].format(**class_context)
        )
        
        previous_class = self.current_class
        self.current_class = class_info
        
        for child in node.body:
            self.visit(child)
            
        self.current_class = previous_class
        self.module.classes[node.name] = class_info

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        raw_code = self.get_node_source(node)
        func_rels = self.relationships[node.name]
        template_key = 'method' if self.current_class else 'function'
        # Format function context properly
        func_context = {
            'path': Path(self.file_path).relative_to(self.repo_path),
            'raw_code': raw_code,
            'defined_in': ".".join(func_rels.defined_in.split('.')[:-1]) if func_rels.defined_in else "None",
            'calls': list(func_rels.calls) if func_rels.calls else "None",
            'reads': list(func_rels.reads_from) if func_rels.reads_from else "None",
            'writes': list(func_rels.writes_to) if func_rels.writes_to else "None",
            'raises': list(func_rels.raises) if func_rels.raises else "None",
            'catches': list(func_rels.catches) if func_rels.catches else "None",
            'type_hints': self._format_dict(func_rels.type_hints),
            'node_map': self.node_map,
            'human_readable_tree': truncate_above_and_below_a_string(self.human_readable_tree, str(Path(self.file_path).relative_to(self.repo_path))),
        }
        
        parameters = []
        for arg in node.args.args:
            param = Parameter(
                name=arg.arg,
                annotation=self._get_annotation_str(arg.annotation),
                default_value=self._get_value_str(arg.default) if hasattr(arg, 'default') else None
            )
            parameters.append(param)
        
        func_info = FunctionInfo(
            name=node.name,
            parameters=parameters,
            return_annotation=self._get_annotation_str(node.returns),
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            docstring=ast.get_docstring(node),
            is_method=self.current_class is not None,
            is_static='@staticmethod' in [ast.unparse(dec) for dec in node.decorator_list],
            is_class_method='@classmethod' in [ast.unparse(dec) for dec in node.decorator_list],
            is_property='@property' in [ast.unparse(dec) for dec in node.decorator_list],
            raw_code=raw_code,
            prompt=PROMPT_TEMPLATES[template_key].format(**func_context)
        )

        previous_function = self.current_function
        self.current_function = func_info
        
        for child in node.body:
            self.visit(child)
            
        self.current_function = previous_function
        
        if self.current_class:
            self.current_class.methods[node.name] = func_info
        else:
            self.module.functions[node.name] = func_info

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if self.current_function:
                self.current_function.calls.append(node.func.id)
            if node.func.id and node.func.id[0].isupper():
                self.module.class_instantiations.append({
                    'class': node.func.id,
                    'args': [ast.unparse(arg) for arg in node.args],
                    'kwargs': [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]
                })
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and self.current_function:
            self.current_function.reads_from.append(node.id)
        elif isinstance(node.ctx, ast.Store) and self.current_function:
            self.current_function.assigns_to.append(node.id)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.current_class and not self.current_function:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if name.isupper():
                        self.module.constants[name] = ast.unparse(node.value)
                    else:
                        self.module.globals[name] = ast.unparse(node.value)
        elif self.current_class and not self.current_function:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.current_class.class_variables[target.id] = ast.unparse(node.value)
        self.generic_visit(node)

class RepoAnalyzer:
    def __init__(self, repo_path: str):
        """
        Initialize the repository analyzer
        
        Args:
            repo_path: Path to the repository root directory
        """
        self.repo_path = Path(repo_path)
        self.modules: Dict[str, ModuleInfo] = {}
        self.human_readable_tree = extract_human_readable_file_tree(self.repo_path)
    
    def analyze_file(self, file_path: str) -> ModuleInfo:
        """
        Analyze a single Python file and generate LLM analysis for its components
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            ModuleInfo object containing the analysis
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        # Get relationships first
        relationships = analyze_relationships(source)
        
        # Pass relationships to AST analyzer
        analyzer = ASTAnalyzer(source, file_path, relationships, self.repo_path, self.human_readable_tree)
        
        try:
            analyzer.visit(ast.parse(source))
            module_info = analyzer.module
        except SyntaxError as e:
            module_info = ModuleInfo(path=file_path, raw_code=source, error=str(e))
        
        # Get LLM analysis for module with relationships context
        if module_info.prompt:
            initial, summary = get_two_stage_analysis(module_info.prompt, "module")
            module_info.llm_analysis = AnalysisResults(initial, summary)
        
        # Get LLM analysis for functions
        for func_info in module_info.functions.values():
            if func_info.prompt:
                initial, summary = get_two_stage_analysis(func_info.prompt, "function")
                func_info.llm_analysis = AnalysisResults(initial, summary)
                
        # Get LLM analysis for classes and methods
        for class_info in module_info.classes.values():
            if class_info.prompt:
                initial, summary = get_two_stage_analysis(class_info.prompt, "class")
                class_info.llm_analysis = AnalysisResults(initial, summary)
                
            for method_info in class_info.methods.values():
                if method_info.prompt:
                    initial, summary = get_two_stage_analysis(method_info.prompt, "method")
                    method_info.llm_analysis = AnalysisResults(initial, summary)
        
        self.modules[file_path] = module_info
        return module_info
    
    def analyze_directory(self, directory: Optional[str] = None) -> Dict[str, ModuleInfo]:
        """
        Recursively analyze all Python files in a directory
        
        Args:
            directory: Directory to analyze, defaults to repo root
            
        Returns:
            Dictionary mapping file paths to ModuleInfo objects
        """
        if directory is None:
            directory = self.repo_path
        else:
            directory = Path(directory)
            
        for path in directory.rglob('*.py'):
            if not path.name.startswith('_') and path.is_file():
                self.analyze_file(str(path))
                
        return self.modules
    
    def save_analysis(self, output_path: str) -> None:
        """
        Save the analysis results to a JSON file
        
        Args:
            output_path: Path to save the JSON output
        """
        output = {
            path: module.to_compact() 
            for path, module in self.modules.items()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
    
    def get_all_functions(self) -> List[FunctionInfo]:
        """
        Get a list of all functions found in the analyzed files
        
        Returns:
            List of FunctionInfo objects
        """
        functions = []
        for module in self.modules.values():
            functions.extend(module.functions.values())
            for class_info in module.classes.values():
                functions.extend(class_info.methods.values())
        return functions
    
    def get_all_classes(self) -> List[ClassInfo]:
        """
        Get a list of all classes found in the analyzed files
        
        Returns:
            List of ClassInfo objects
        """
        classes = []
        for module in self.modules.values():
            classes.extend(module.classes.values())
        return classes
    
    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Generate a dependency graph of modules based on imports
        
        Returns:
            Dictionary mapping module paths to sets of imported module paths
        """
        graph = {}
        for path, module in self.modules.items():
            dependencies = set()
            for imp_type, imports in module.imports.items():
                for imp in imports:
                    # Convert import to potential file path
                    imp_parts = imp.split('.')
                    potential_path = str(self.repo_path / '/'.join(imp_parts) + '.py')
                    if potential_path in self.modules:
                        dependencies.add(potential_path)
            graph[path] = dependencies
        return graph

if __name__ == "__main__":
    #model = AutoModelForCausalLM.from_pretrained(
    #    model_name,
    #    torch_dtype=torch.float16,
    #    device_map="auto"
    #)
    model = LLM(model=model_name,trust_remote_code=True)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    if len(sys.argv) < 2:
        repo_path = THIS_DIR
    else:
        repo_path = sys.argv[1]
    
    analyzer = RepoAnalyzer(repo_path)
    analyzer.analyze_directory()
    analyzer.save_analysis(THIS_DIR / 'analysis.json')
