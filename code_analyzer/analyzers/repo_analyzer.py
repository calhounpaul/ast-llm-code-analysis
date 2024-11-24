# ast-llm-code-analysis/code_analyzer/analyzers/repo_analyzer.py

from pathlib import Path
import ast
import json
from typing import Dict, List, Optional, Set
from collections import defaultdict

from ..models.data_classes import ModuleInfo, ClassInfo, FunctionInfo, AnalysisResults
from ..analyzers.ast_analyzer import ASTAnalyzer
from ..analyzers.relationship_visitor import analyze_relationships
from ..utils.text_processing import extract_human_readable_file_tree
from ..utils.llm_query import LLMQueryManager

class RepoAnalyzer:
    """Analyzes Python repositories to extract code structure and relationships."""
    
    def __init__(self, repo_path: str, llm_manager: Optional[LLMQueryManager] = None):
        """
        Initialize the repository analyzer.
        
        Args:
            repo_path: Path to the repository root directory
            llm_manager: LLMQueryManager instance for queries (creates default if None)
        """
        self.repo_path = Path(repo_path)
        self.modules: Dict[str, ModuleInfo] = {}
        self.human_readable_tree = extract_human_readable_file_tree(self.repo_path)
        self.llm_manager = llm_manager or LLMQueryManager()
    
    def get_analysis(self, prompt: str, entity_type: str) -> tuple[str, str]:
        """
        Get two-stage analysis using the configured LLMQueryManager.
        
        Args:
            prompt: The analysis prompt
            entity_type: Type of entity being analyzed
            
        Returns:
            Tuple of (initial analysis, summary)
        """
        # First stage: Initial analysis
        initial_analysis = self.llm_manager.query(prompt)
        
        # Second stage: Summary
        summary_prompt = f"""
        Excellent work. Please provide a succinct summary of this {entity_type} in the following format:
        SUMMARY: 3-sentence summary goes here
        NOTES: any other comments or notes go here
        """.strip()
        
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial_analysis},
            {"role": "user", "content": summary_prompt}
        ]
        
        summary = self.llm_manager.query_multi_turn(messages)
        
        return initial_analysis, summary
    
    def analyze_file(self, file_path: str) -> ModuleInfo:
        """Analyze a single Python file and generate LLM analysis for its components."""
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        # Extract code relationships
        relationships = analyze_relationships(source)
        
        # Perform AST analysis
        analyzer = ASTAnalyzer(
            source=source,
            file_path=file_path,
            relationships=relationships,
            repo_path=self.repo_path,
            human_readable_tree=self.human_readable_tree,
            llm_manager=self.llm_manager  # Pass the manager to ASTAnalyzer
        )
        
        try:
            analyzer.visit(ast.parse(source))
            module_info = analyzer.module
        except SyntaxError as e:
            module_info = ModuleInfo(path=file_path, raw_code=source, error=str(e))
            return module_info
        
        # Generate LLM analysis for module
        if module_info.prompt:
            initial, summary = self.get_analysis(module_info.prompt, "module")
            module_info.llm_analysis = AnalysisResults(initial, summary)
        
        # Generate LLM analysis for functions
        for func_info in module_info.functions.values():
            if func_info.prompt:
                initial, summary = self.get_analysis(func_info.prompt, "function")
                func_info.llm_analysis = AnalysisResults(initial, summary)
                
        # Generate LLM analysis for classes and methods
        for class_info in module_info.classes.values():
            if class_info.prompt:
                initial, summary = self.get_analysis(class_info.prompt, "class")
                class_info.llm_analysis = AnalysisResults(initial, summary)
                
            for method_info in class_info.methods.values():
                if method_info.prompt:
                    initial, summary = self.get_analysis(method_info.prompt, "method")
                    method_info.llm_analysis = AnalysisResults(initial, summary)
        
        self.modules[file_path] = module_info
        return module_info

class RepoAnalyzer:
    """Analyzes Python repositories to extract code structure and relationships."""
    
    def __init__(self, repo_path: str, llm_manager: Optional[LLMQueryManager] = None):
        """
        Initialize the repository analyzer.
        
        Args:
            repo_path: Path to the repository root directory
            llm_manager: LLMQueryManager instance for queries (creates default if None)
        """
        self.repo_path = Path(repo_path)
        self.modules: Dict[str, ModuleInfo] = {}
        self.human_readable_tree = extract_human_readable_file_tree(self.repo_path)
        self.llm_manager = llm_manager or LLMQueryManager()
    
    def get_analysis(self, prompt: str, entity_type: str) -> tuple[str, str]:
        """
        Get two-stage analysis using the configured LLMQueryManager.
        
        Args:
            prompt: The analysis prompt
            entity_type: Type of entity being analyzed
            
        Returns:
            Tuple of (initial analysis, summary)
        """
        # First stage: Initial analysis
        initial_analysis = self.llm_manager.query(prompt)
        
        # Second stage: Summary
        summary_prompt = f"""
        Excellent work. Please provide a succinct summary of this {entity_type} in the following format:
        SUMMARY: 3-sentence summary goes here
        NOTES: any other comments or notes go here
        """.strip()
        
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial_analysis},
            {"role": "user", "content": summary_prompt}
        ]
        
        summary = self.llm_manager.query_multi_turn(messages)
        
        return initial_analysis, summary
    
    def analyze_file(self, file_path: str) -> ModuleInfo:
        """Analyze a single Python file and generate LLM analysis for its components."""
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        # Extract code relationships
        relationships = analyze_relationships(source)
        
        # Perform AST analysis
        analyzer = ASTAnalyzer(
            source=source,
            file_path=file_path,
            relationships=relationships,
            repo_path=self.repo_path,
            human_readable_tree=self.human_readable_tree,
            llm_manager=self.llm_manager  # Pass the manager to ASTAnalyzer
        )
        
        try:
            analyzer.visit(ast.parse(source))
            module_info = analyzer.module
        except SyntaxError as e:
            module_info = ModuleInfo(path=file_path, raw_code=source, error=str(e))
            return module_info
        
        # Generate LLM analysis for module
        if module_info.prompt:
            initial, summary = self.get_analysis(module_info.prompt, "module")
            module_info.llm_analysis = AnalysisResults(initial, summary)
        
        # Generate LLM analysis for functions
        for func_info in module_info.functions.values():
            if func_info.prompt:
                initial, summary = self.get_analysis(func_info.prompt, "function")
                func_info.llm_analysis = AnalysisResults(initial, summary)
                
        # Generate LLM analysis for classes and methods
        for class_info in module_info.classes.values():
            if class_info.prompt:
                initial, summary = self.get_analysis(class_info.prompt, "class")
                class_info.llm_analysis = AnalysisResults(initial, summary)
                
            for method_info in class_info.methods.values():
                if method_info.prompt:
                    initial, summary = self.get_analysis(method_info.prompt, "method")
                    method_info.llm_analysis = AnalysisResults(initial, summary)
        
        self.modules[file_path] = module_info
        return module_info

    def analyze_directory(self, directory: Optional[str] = None) -> Dict[str, ModuleInfo]:
        """
        Recursively analyze all Python files in a directory.
        
        Args:
            directory: Directory to analyze, defaults to repo root
            
        Returns:
            Dictionary mapping file paths to ModuleInfo objects
        """
        if directory is None:
            directory = self.repo_path
        else:
            directory = Path(directory)
            
        print("Directory being analyzed:", directory)
        
        for path in directory.rglob('*.py'):
            print("Found file:", path)
            # Skip hidden files and test files
            if not path.name.startswith('_') and not path.name.startswith('test_') and path.is_file():
                print("Analyzing file:", path)
                self.analyze_file(str(path))
                    
        return self.modules

    
    def save_analysis(self, output_path: str) -> None:
        """
        Save the analysis results to a JSON file.
        
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
        Get a list of all functions found in the analyzed files.
        
        Returns:
            List of FunctionInfo objects for all functions and methods
        """
        functions = []
        for module in self.modules.values():
            functions.extend(module.functions.values())
            for class_info in module.classes.values():
                functions.extend(class_info.methods.values())
        return functions
    
    def get_all_classes(self) -> List[ClassInfo]:
        """
        Get a list of all classes found in the analyzed files.
        
        Returns:
            List of ClassInfo objects
        """
        classes = []
        for module in self.modules.values():
            classes.extend(module.classes.values())
        return classes
    
    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Generate a dependency graph of modules based on imports.
        
        The graph shows which modules depend on other modules based on
        their import statements.
        
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

    def get_module_statistics(self) -> Dict[str, int]:
        """
        Calculate basic statistics about the analyzed codebase.
        
        Returns:
            Dictionary containing counts of various code elements:
            - total_modules: Number of Python files analyzed
            - total_classes: Number of classes found
            - total_functions: Number of standalone functions
            - total_methods: Number of class methods
            - total_lines: Approximate total lines of code
        """
        stats = {
            'total_modules': len(self.modules),
            'total_classes': sum(len(m.classes) for m in self.modules.values()),
            'total_functions': sum(len(m.functions) for m in self.modules.values()),
            'total_methods': sum(
                sum(len(c.methods) for c in m.classes.values())
                for m in self.modules.values()
            ),
            'total_lines': sum(
                m.raw_code.count('\n') + 1 for m in self.modules.values()
            )
        }
        return stats