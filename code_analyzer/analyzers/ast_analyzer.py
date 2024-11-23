# ast-llm-code-analysis/code_analyzer/analyzers/ast_analyzer.py

import ast
from pathlib import Path
from typing import Dict, Optional, Any, Union, List, Set

from code_analyzer.models.data_classes import (
    ModuleInfo, ClassInfo, FunctionInfo, Parameter, AnalysisResults
)
from code_analyzer.utils.text_processing import truncate_middle, truncate_above_and_below_a_string
from code_analyzer.utils.llm_query import get_two_stage_analysis, LLMQueryManager
from code_analyzer.analyzers.relationship_visitor import NodeRelationships
from code_analyzer.utils.prompt_loader import load_prompt_templates

def extract_map_of_nodes(source: str, max_depth: int = 3) -> Dict[str, Any]:
    """
    Extract a map of functions, classes, imports, and other nodes from source code.
    
    Args:
        source: Source code to analyze
        max_depth: Maximum depth of node traversal
        
    Returns:
        Dictionary mapping node types to lists of node names
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        if "f-string expression part cannot include a backslash" in str(e):
            source = source.replace("\\", "")
            tree = ast.parse(source)
        else:
            raise e
            
    node_map = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_type = node.__class__.__name__
            if node_type not in node_map:
                node_map[node_type] = []
            node_map[node_type].append(node.name)
            
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            node_type = node.__class__.__name__
            if node_type not in node_map:
                node_map[node_type] = []
            node_map[node_type].extend(alias.name for alias in node.names)
            
        if len(node_map) >= max_depth:
            break
            
    return node_map

class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor that analyzes Python source code and builds module, class, and function info."""
    
    def __init__(
        self, 
        source: str,
        file_path: str,
        relationships: Dict[str, NodeRelationships],
        repo_path: Path,
        human_readable_tree: str,
        llm_manager: Optional[LLMQueryManager] = None
    ):
        """Initialize the AST analyzer."""
        self.source = source
        self.repo_path = repo_path
        self.file_path = file_path
        self.relationships = relationships
        self.current_class: Optional[ClassInfo] = None
        self.current_function: Optional[FunctionInfo] = None
        self.module = ModuleInfo(path=file_path, raw_code=source)
        self.node_map = truncate_middle(str(extract_map_of_nodes(source)), max_length=600)
        self.human_readable_tree = human_readable_tree
        self.llm_manager = llm_manager or LLMQueryManager()
        self.prompt_templates = load_prompt_templates()

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
        """Extract source code for a given AST node"""
        lines = self.source.splitlines()
        return '\n'.join(lines[node.lineno - 1:node.end_lineno])

    def _get_annotation_str(self, annotation: Optional[ast.AST]) -> Optional[str]:
        """Get string representation of a type annotation"""
        if annotation is None:
            return None
        return ast.unparse(annotation)

    def _get_value_str(self, value: Optional[ast.AST]) -> Optional[str]:
        """Get string representation of an AST value"""
        if value is None:
            return None
        return ast.unparse(value)

    def visit_Module(self, node: ast.Module) -> None:
        """Process module-level nodes and generate module analysis"""
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
            'node_map': self.node_map,
            'human_readable_tree': truncate_above_and_below_a_string(
                self.human_readable_tree, 
                str(Path(self.file_path).relative_to(self.repo_path))
            ),
        }

        self.module.prompt = self.prompt_templates['module'].format(**module_context)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Process class definitions and generate class analysis"""
        raw_code = self.get_node_source(node)
        class_rels = self.relationships[node.name]
        
        class_context = {
            'path': Path(self.file_path).relative_to(self.repo_path),
            'raw_code': raw_code,
            'defined_in': class_rels.defined_in or "None",
            'inherits': self._format_list(class_rels.inherits_from),
            'decorators': self._format_list(class_rels.calls),
            'contains': self._format_list(class_rels.contains),
            'imports': self._format_list(class_rels.imports),
            'node_map': self.node_map,
            'human_readable_tree': truncate_above_and_below_a_string(
                self.human_readable_tree,
                str(Path(self.file_path).relative_to(self.repo_path))
            ),
        }
        
        class_info = ClassInfo(
            name=node.name,
            bases=[ast.unparse(base) for base in node.bases],
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            docstring=ast.get_docstring(node),
            raw_code=raw_code,
            prompt=self.prompt_templates['class'].format(**class_context)
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definitions and generate function/method analysis"""
        raw_code = self.get_node_source(node)
        func_rels = self.relationships[node.name]
        template_key = 'method' if self.current_class else 'function'
        
        func_context = {
            'path': Path(self.file_path).relative_to(self.repo_path),
            'raw_code': raw_code,
            'defined_in': ".".join(func_rels.defined_in.split('.')[:-1]) if func_rels.defined_in else "None",
            'calls': self._format_list(func_rels.calls),
            'reads': self._format_list(func_rels.reads_from),
            'writes': self._format_list(func_rels.writes_to),
            'raises': self._format_list(func_rels.raises),
            'catches': self._format_list(func_rels.catches),
            'type_hints': self._format_dict(func_rels.type_hints),
            'node_map': self.node_map,
            'human_readable_tree': truncate_above_and_below_a_string(
                self.human_readable_tree,
                str(Path(self.file_path).relative_to(self.repo_path))
            ),
        }
        
        parameters = [
            Parameter(
                name=arg.arg,
                annotation=self._get_annotation_str(arg.annotation),
                default_value=self._get_value_str(arg.default) if hasattr(arg, 'default') else None
            )
            for arg in node.args.args
        ]
        
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
            prompt=self.prompt_templates[template_key].format(**func_context)
        )

        previous_function = self.current_function
        self.current_function = func_info
        self.generic_visit(node)
        self.current_function = previous_function
        
        if self.current_class:
            self.current_class.methods[node.name] = func_info
        else:
            self.module.functions[node.name] = func_info

    def analyze(self) -> ModuleInfo:
        """
        Perform the AST analysis and return the results.
        
        Returns:
            ModuleInfo object containing the analysis results
        """
        try:
            self.visit(ast.parse(self.source))
            
            # Get LLM analysis for module
            if self.module.prompt:
                initial, summary = get_two_stage_analysis(self.module.prompt, "module")
                self.module.llm_analysis = AnalysisResults(initial, summary)
            
            # Get LLM analysis for functions
            for func_info in self.module.functions.values():
                if func_info.prompt:
                    initial, summary = get_two_stage_analysis(func_info.prompt, "function")
                    func_info.llm_analysis = AnalysisResults(initial, summary)
                    
            # Get LLM analysis for classes and methods
            for class_info in self.module.classes.values():
                if class_info.prompt:
                    initial, summary = get_two_stage_analysis(class_info.prompt, "class")
                    class_info.llm_analysis = AnalysisResults(initial, summary)
                    
                for method_info in class_info.methods.values():
                    if method_info.prompt:
                        initial, summary = get_two_stage_analysis(method_info.prompt, "method")
                        method_info.llm_analysis = AnalysisResults(initial, summary)
                        
        except SyntaxError as e:
            self.module.error = str(e)
            
        return self.module