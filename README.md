# AI-Powered Code Analysis Tool

## Overview
This tool performs intelligent code analysis by combining Abstract Syntax Tree (AST) parsing with the Qwen2.5 large language model to generate comprehensive code documentation and insights. It analyzes Python codebases at multiple levels (modules, classes, functions) and generates both detailed analysis and summaries.

## Key Features

### Static Analysis
- Parses codebase using Python's AST
- Tracks function calls, variable usage, and dependencies
- Identifies relationships between code components:
  - Function calls and dependencies
  - Variable reads and modifications
  - Class inheritance patterns
  - Import relationships
  - Type hints and annotations

### AI-Powered Analysis
- Uses Qwen2.5-Coder-14B-Instruct-AWQ model
- Generates two-stage analysis for each code component:
  1. Detailed Analysis
     - Purpose and responsibilities
     - Implementation details
     - Dependency interactions
     - Error handling patterns
     - Edge cases consideration
  2. Concise Summary
     - 3-sentence overview
     - Additional notes/comments

### Analysis Context
For each code component, provides rich context including:
- File path and location
- Code relationships (calls, imports, etc.)
- Variable usage patterns
- Exception handling
- Type information
- Repository structure context

## Deployment

### Docker Environment
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# ... (previous Docker configuration)
```

Requirements:
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- 16GB+ GPU memory recommended

### Quick Start
```bash
# Build container
docker build -t code-analyzer .

# Analyze a codebase
docker run --gpus all \
  -v /path/to/workdir/with/script:/home/mluser/workdir \
  -v /path/to/cache/folder:/home/mluser/.cache \
  -w /home/mluser/workdir \
  code-analyzer

# You should get a bash prompt.
# Just run the python script with the sole argument as the root path of the tree you'd like to analyze.
# The output ends up in analysis.json
```

### Output Format
```json
{
  "path/to/file.py": {
    "functions": {
      "function_name": {
        "analysis": "Detailed function analysis...",
        "summary": {
          "SUMMARY": "3-sentence summary...",
          "NOTES": "Additional insights..."
        }
      }
    },
    "classes": { ... },
    "modules": { ... }
  }
}
```

## Performance Optimizations
- SQLite-based response caching
- Efficient text truncation for large files
- Base64 encoding for cache storage
- Debug logging capabilities
- GPU acceleration support

## Use Cases
- Code documentation generation
- Technical debt analysis
- Developer onboarding
- Architecture documentation
- Code quality assessment
- Pattern recognition

## Limitations
- Requires CUDA-capable GPU
- Analysis depth limited by model context window
- Python-specific analysis only
- May require manual verification for complex code patterns
