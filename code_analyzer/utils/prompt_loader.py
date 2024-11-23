# ast-llm-code-analysis/code_analyzer/utils/prompt_loader.py

import json
from pathlib import Path
from typing import Dict, Any

def load_prompt_templates() -> Dict[str, str]:
    """
    Load prompt templates from prompts.json file.
    
    Returns:
        Dictionary mapping template names to their content
        
    Raises:
        FileNotFoundError: If prompts.json is not found
        KeyError: If expected template structure is not found
    """
    prompts_path = Path(__file__).parent.parent.parent / 'assets' / 'prompts.json'
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Extract just the template strings from the loaded data
    templates = {}
    for template_name, template_data in data['templates'].items():
        templates[template_name] = template_data['template'].strip()
    
    return templates