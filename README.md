# Code Analyzer

A powerful Python code analysis tool that leverages Large Language Models (LLM) and Abstract Syntax Tree (AST) analysis to provide comprehensive insights into Python codebases. The analyzer generates detailed analysis of code structure, relationships, and provides AI-powered summaries using the Qwen 2.5 model.

## Features

- 🔍 Deep code analysis using Python's AST
  - Function and method analysis
  - Class hierarchy and relationships
  - Module dependencies and imports
  - Variable usage and modifications
  - Type hint extraction
  - Exception handling patterns

- 🤖 AI-powered code understanding
  - Comprehensive code summaries using Qwen 2.5
  - Intelligent relationship mapping
  - Context-aware analysis
  - Multi-turn analysis generation

- 💾 Performance optimizations
  - Efficient caching of LLM responses using SQLite
  - Clean separation of analysis and LLM components
  - Support for large codebases
  - Debug logging and export capabilities

- 📊 Analysis outputs
  - Detailed JSON reports
  - Human-readable summaries
  - Dependency graphs
  - Code statistics

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (for LLM inference)
- Docker (optional, for containerized execution)

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd code-analyzer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Docker Installation

```bash
# Build the Docker image
./run_in_docker.sh /path/to/analyze
```

## Usage

### Command Line Interface

```bash
# Basic analysis
code-analyzer /path/to/your/repo

# Analysis with additional options
code-analyzer /path/to/your/repo \
  --output analysis.json \
  --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
  --debug \
  --verbose \
  --export
```

### Python API

```python
from code_analyzer.analyzers.repo_analyzer import RepoAnalyzer

# Initialize analyzer
analyzer = RepoAnalyzer("/path/to/your/repo")

# Analyze repository
results = analyzer.analyze_directory()

# Get analysis components
functions = analyzer.get_all_functions()
classes = analyzer.get_all_classes()
dependencies = analyzer.get_dependency_graph()
stats = analyzer.get_module_statistics()

# Save analysis
analyzer.save_analysis("analysis.json")
```

## Configuration

The analyzer supports several environment variables for configuration:

- `CODE_ANALYZER_CACHE_DIR`: Directory for SQLite cache
- `CODE_ANALYZER_LOG_DIR`: Directory for log files
- `CODE_ANALYZER_LLM_DIR`: Directory for LLM query exports
- `CODE_ANALYZER_DEBUG_DIR`: Directory for debug files

## Output Format

The analyzer generates a structured JSON output:

```json
{
  "file_path.py": {
    "path": "file_path.py",
    "classes": {
      "ClassName": {
        "name": "ClassName",
        "methods": {...},
        "llm_analysis": {
          "initial_analysis": "...",
          "summary": "..."
        }
      }
    },
    "functions": {...},
    "imports": {...}
  }
}
```

## Development

### Project Structure

```
code_analyzer/
├── code_analyzer/
│   ├── analyzers/      # Core analysis components
│   ├── models/         # Data models and cache management
│   └── utils/          # Helper utilities
├── assets/            # Configuration and prompts
├── tests/             # Test suite
└── docker/            # Docker configuration
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=code_analyzer tests/
```

## Limitations

- Currently optimized for Python code analysis only
- Requires access to Qwen model for LLM analysis
- Large repositories may require significant processing time
- Cache size can grow significantly for large codebases

## License

This project is licensed under the MIT License - see the LICENSE file for details.