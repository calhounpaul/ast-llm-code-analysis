# Code Analyzer

A Python code analysis tool that leverages Large Language Models (LLM) and Abstract Syntax Tree (AST) analysis to provide comprehensive insights into Python codebases. The analyzer generates detailed analysis of code structure, relationships, and provides AI-powered summaries using any instruct model [supported by vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Current Features

### Static Analysis
- AST-based code structure analysis
- Function and method analysis
- Class hierarchy mapping
- Variable usage analysis
- Type hint extraction
- Exception handling analysis

### AI-Powered Understanding
- Contextual code analysis using Qwen 2.5 models
- Two-turn analysis generation
- Module and function level summaries
- Basic code relationship detection

### Performance & Reliability
- SQLite-based LLM response caching
- GPU acceleration support via CUDA
- Comprehensive error handling
- Basic logging capabilities

### Output & Visualization
- Structured JSON reports
- Basic contextual summaries
- Debug logs
- Simple analysis statistics

## Future Features (largely hallucinated by Claude 3.5 )

### Enhanced Analysis
- Advanced dependency and relationship mapping
- Comprehensive call graph generation
- Deep exception flow tracking
- Semantic search capabilities

### Advanced AI Features
- Multi-turn analysis generation (beyond 2 turns)
- Hierarchical summaries at all levels
- Intelligent relationship mapping
- Advanced context understanding

### DevOps & Team Tools
- Containerized execution environment
- Full stack visualization
- Team harmonization tools
- Cloud optimization features
- Exception solution resolution

### Extended Visualizations
- Interactive dependency graphs
- Advanced code statistics
- Rich visual reports
- Extended analysis exports

## Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA support
- Docker (recommended)

### Docker Installation (Recommended)
```bash
# Clone the repository
git clone <repository-url>

# Run analysis using Docker
./run_in_docker.sh /path/to/analyze --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### Command Line Interface
```bash
# Basic analysis
code-analyzer /path/to/analyze

# Advanced options
code-analyzer /path/to/analyze \
    --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
    --output analysis.json \
    --export \
    --verbose \
    --debug
```

### Python API
```python
from code_analyzer.analyzers.repo_analyzer import RepoAnalyzer
from code_analyzer.utils.llm_query import LLMQueryManager

# Initialize LLM manager
llm_manager = LLMQueryManager(
    verbose=True,
    use_cache=True,
    export=True
)

# Create analyzer instance
analyzer = RepoAnalyzer(
    repo_path="/path/to/analyze",
    llm_manager=llm_manager
)

# Run analysis
modules = analyzer.analyze_directory()

# Get analysis components
stats = analyzer.get_module_statistics()
functions = analyzer.get_all_functions()
classes = analyzer.get_all_classes()
dependencies = analyzer.get_dependency_graph()

# Save results
analyzer.save_analysis("analysis.json")
```

## Configuration

### Environment Variables
```
CODE_ANALYZER_CACHE_DIR  # SQLite cache location
CODE_ANALYZER_LOG_DIR    # Log file directory
CODE_ANALYZER_LLM_DIR    # LLM query export directory
CODE_ANALYZER_DEBUG_DIR  # Debug file location
```

### Project Structure
```
code_analyzer/
├── analyzers/          # Core analysis components
│   ├── ast_analyzer.py     # AST parsing and analysis
│   ├── relationship_visitor.py  # Code relationship analysis
│   └── repo_analyzer.py    # Repository-level analysis
├── models/             # Data structures
│   ├── data_classes.py     # Core data models
│   └── cache_manager.py    # LLM response caching
├── utils/              # Supporting utilities
└── assets/            # Configuration files and prompts
```

## Output Format

The analyzer generates detailed JSON reports containing:
- Module-level analysis
- Class hierarchies and relationships
- Function and method details
- AI-generated summaries
- Code statistics and metrics

Example output structure:
```json
{
  "file_path.py": {
    "path": "file_path.py",
    "classes": {
      "ClassName": {
        "name": "ClassName",
        "bases": ["ParentClass"],
        "methods": {},
        "llm_analysis": {
          "initial_analysis": "...",
          "summary": "..."
        }
      }
    },
    "functions": {},
    "imports": {},
    "error": null
  }
}
```

## Development

### Running Tests
```bash
# Run all tests
./run_in_docker.sh --test

# Run specific test file
./run_in_docker.sh --test tests/test_ast_analyzer.py

# Run with coverage
./run_in_docker.sh --test --pytest-args '--cov=code_analyzer/ --cov-report=html'
```

## Limitations

- Python-specific analysis only
- Requires NVIDIA GPU for optimal performance
- Large repositories may require significant processing time
- Cache storage requirements scale with codebase size
- Relies on external Qwen model availability

## License

This project is licensed under the MIT License. See the LICENSE file for details.
