#ast-llm-code-analysis/code_analyzer/utils/text_processing.py

import os
from pathlib import Path
from typing import Optional
from datetime import datetime
import hashlib
import textwrap

# Get the module's directory
THIS_DIR = Path(__file__).resolve().parent

# Debug logging configuration
DEBUG_TO_TXT = True  # Can be configured via environment variable

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
        
    debug_dir = THIS_DIR.parent / 'debug'
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

def truncate_middle(s: str, max_length: int = 4000) -> str:
    """
    Truncate a string in the middle to a maximum length while preserving content at both ends.
    
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

def extract_human_readable_file_tree(repo_path: Path) -> str:
    """
    Extract a human-readable file tree structure for a given folder.
    
    Args:
        repo_path: Path object pointing to the repository root
        
    Returns:
        String representation of the file tree with /base/ prefix
    """
    tree = []
    for path in repo_path.rglob('*'):
        if path.is_file():
            tree.append("/base/" + str(path.relative_to(repo_path)))
    return "\n".join(sorted(tree))

def get_node_source(source: str, node) -> str:
    """
    Extract the source code for a given AST node.
    
    Args:
        source: Complete source code string
        node: AST node with lineno and end_lineno attributes
        
    Returns:
        Source code corresponding to the node
    """
    lines = source.splitlines()
    return '\n'.join(lines[node.lineno - 1:node.end_lineno])

def format_list(items: Optional[list]) -> str:
    """
    Format a list of items for display.
    
    Args:
        items: List of items to format, or None
        
    Returns:
        Comma-separated string of items, or "None" if empty/None
    """
    if not items:
        return "None"
    return ", ".join(sorted(str(item) for item in items))

def format_dict(d: Optional[dict]) -> str:
    """
    Format a dictionary for display.
    
    Args:
        d: Dictionary to format, or None
        
    Returns:
        Formatted string representation of dictionary items, or "None" if empty/None
    """
    if not d:
        return "None"
    return ", ".join(f"{k}: {v}" for k, v in sorted(d.items()))