#ast-llm-code-analysis/code_analyzer/analyzers/relationship_visitor.py

import ast
from typing import Dict, Set, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class NodeRelationships:
    """
    Tracks relationships between AST nodes including dependencies, variables,
    structural relationships, type information, and control flow.
    """
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
        """
        Merge another NodeRelationships object into this one.
        
        Args:
            other: Another NodeRelationships object to merge into this one
        """
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
    """
    AST visitor that analyzes relationships between Python code elements.
    Tracks function calls, variable usage, inheritance, imports, and more.
    """
    
    def __init__(self):
        """Initialize the visitor with empty relationship tracking."""
        self.relationships: Dict[str, NodeRelationships] = defaultdict(NodeRelationships)
        self.scope_stack: List[str] = []
    
    @property
    def current_scope_name(self) -> str:
        """Get the fully qualified name of the current scope."""
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visit a class definition, tracking inheritance, decorators, and scope.
        
        Args:
            node: AST node for class definition
        """
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
        """
        Visit a function definition, tracking return types, parameters, and scope.
        
        Args:
            node: AST node for function definition
        """
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
        """
        Visit a function/method call, tracking the called names.
        
        Args:
            node: AST node for function call
        """
        if self.scope_stack:
            caller = self.relationships[self.scope_stack[-1]]
            if isinstance(node.func, ast.Name):
                caller.calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                caller.calls.add(ast.unparse(node.func))
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        """
        Visit a name node, tracking variable reads and writes.
        
        Args:
            node: AST node for name
        """
        if self.scope_stack:
            current = self.relationships[self.scope_stack[-1]]
            if isinstance(node.ctx, ast.Load):
                current.reads_from.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                current.writes_to.add(node.id)
        self.generic_visit(node)
    
    def visit_Raise(self, node: ast.Raise) -> None:
        """
        Visit a raise statement, tracking raised exceptions.
        
        Args:
            node: AST node for raise statement
        """
        if self.scope_stack and node.exc:
            current = self.relationships[self.scope_stack[-1]]
            if isinstance(node.exc, ast.Name):
                current.raises.add(node.exc.id)
            elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                current.raises.add(node.exc.func.id)
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """
        Visit an except handler, tracking caught exceptions.
        
        Args:
            node: AST node for except handler
        """
        if self.scope_stack and node.type:
            current = self.relationships[self.scope_stack[-1]]
            if isinstance(node.type, ast.Name):
                current.catches.add(node.type.id)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Visit a from-import statement, tracking imported names.
        
        Args:
            node: AST node for from-import
        """
        if self.scope_stack:
            current = self.relationships[self.scope_stack[-1]]
            module = node.module or ''
            for alias in node.names:
                current.imports.add(f"{module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """
        Visit an import statement, tracking imported modules.
        
        Args:
            node: AST node for import
        """
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
        
    Raises:
        SyntaxError: If the source code contains invalid syntax
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