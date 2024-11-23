#ast-llm-code-analysis/code_analyzer/utils/helpers.py

from pathlib import Path
from typing import Dict, Any, Set, List, Optional
import ast
from datetime import datetime
import hashlib
import base64

def extract_human_readable_file_tree(repo_path: Path) -> str:
    """
    Extract a human-readable file tree structure for a given folder.
    
    Args:
        repo_path: Path object pointing to repository root
        
    Returns:
        String containing formatted file tree
    """
    tree = []
    for path in repo_path.rglob('*'):
        if path.is_file():
            tree.append("/base/"+str(path.relative_to(repo_path)))
    return "\n".join(sorted(tree))

def truncate_middle(s: str, max_length: int = 4000) -> str:
    """
    Truncate a string in the middle to a maximum length while preserving context.
    
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

def truncate_above_and_below_a_string(s: str, target: str, max_chars: int = 600, clean_cut_edges: bool = True) -> str:
    """
    Truncate a string at two locations while preserving a target line and its context.
    
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
    """
    Extract a map of functions, classes, imports, and other nodes from source code.
    
    Args:
        source: Python source code to analyze
        max_depth: Maximum depth of node traversal
    
    Returns:
        Dictionary mapping node types to lists of node names
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        if "f-string expression part cannot include a backslash" in str(e):
            # Remove backslashes from f-strings and try again
            source = source.replace("\\", "")
            tree = ast.parse(source)
        else:
            raise e
            
    from collections import defaultdict
    node_map = defaultdict(list)
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_map[node.__class__.__name__].append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            node_map[node.__class__.__name__].extend(alias.name for alias in node.names)
        if len(node_map) >= max_depth:
            break
            
    return dict(node_map)

def get_query_hash(content: str) -> str:
    """
    Generate a unique hash for given content.
    
    Args:
        content: The input content to hash
        
    Returns:
        SHA-256 hash of the content
    """
    return hashlib.sha256(content.encode()).hexdigest()

def encode_base64(content: str) -> str:
    """
    Encode content as Base64.
    
    Args:
        content: Content to encode
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(content.encode('utf-8')).decode('utf-8')

def decode_base64(content: str) -> str:
    """
    Decode Base64 content.
    
    Args:
        content: Base64 encoded content
        
    Returns:
        Decoded string
        
    Raises:
        ValueError: If content cannot be decoded
    """
    try:
        return base64.b64decode(content).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to decode content: {e}")

def format_timestamp() -> str:
    """
    Generate a formatted timestamp for the current time.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def format_file_path(file_path: Path, repo_path: Path) -> str:
    """
    Format a file path relative to repository root.
    
    Args:
        file_path: Path to the file
        repo_path: Path to repository root
        
    Returns:
        Formatted relative path string
    """
    try:
        return str(file_path.relative_to(repo_path))
    except ValueError:
        return str(file_path)

class CompactOutput:
    """Helper class to manage compact output format for JSON serialization"""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.empty_keys: Set[str] = set()
    
    def add(self, key: str, value: Any) -> None:
        """
        Add a key-value pair to the output, tracking empty values.
        
        Args:
            key: Dictionary key
            value: Value to store
        """
        if value:
            if isinstance(value, dict) and not any(value.values()):
                self.empty_keys.add(key)
            elif isinstance(value, (list, set)) and not value:
                self.empty_keys.add(key)
            else:
                self.data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the output to a dictionary.
        
        Returns:
            Dictionary containing non-empty values and list of empty keys
        """
        if self.empty_keys:
            self.data['empty_keys'] = sorted(list(self.empty_keys))
        return self.data

def format_list(items: Optional[List[str]]) -> str:
    """
    Format a list of items for display.
    
    Args:
        items: List of items to format
        
    Returns:
        Comma-separated string of items or "None"
    """
    if not items:
        return "None"
    return ", ".join(sorted(items))

def format_dict(d: Optional[Dict[str, Any]]) -> str:
    """
    Format a dictionary for display.
    
    Args:
        d: Dictionary to format
        
    Returns:
        Formatted string representation of dictionary items
    """
    if not d:
        return "None"
    return ", ".join(f"{k}: {v}" for k, v in sorted(d.items()))