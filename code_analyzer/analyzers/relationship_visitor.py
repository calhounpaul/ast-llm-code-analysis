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
        """Merge another NodeRelationships object into this one."""
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
        super().__init__()
        self.relationships = defaultdict(NodeRelationships)
        self.scope_stack = ["<module>"]  # Initialize with module scope
    
    @property
    def current_scope_name(self) -> str:
        return self.scope_stack[-1] if self.scope_stack else "<module>"
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Get the fully qualified name based on current scope
        qualified_name = f"{self.current_scope_name}.{node.name}" if self.current_scope_name != "<module>" else node.name
        
        relationships = self.relationships[qualified_name]
        relationships.defined_in = self.current_scope_name
        
        # Track inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                relationships.inherits_from.add(base.id)
            elif isinstance(base, ast.Attribute):
                relationships.inherits_from.add(ast.unparse(base))
        
        # Track decorator calls
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                relationships.calls.add(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    relationships.calls.add(decorator.func.id)
        
        self.scope_stack.append(qualified_name)
        self.generic_visit(node)
        self.scope_stack.pop()
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        qualified_name = f"{self.current_scope_name}.{node.name}" if self.current_scope_name != "<module>" else node.name
        
        relationships = self.relationships[qualified_name]
        relationships.defined_in = self.current_scope_name
        
        # Track return type annotation
        if node.returns:
            relationships.type_hints['return'] = ast.unparse(node.returns)
        
        # Track parameter type hints
        for arg in node.args.args:
            if arg.annotation:
                relationships.type_hints[arg.arg] = ast.unparse(arg.annotation)
        
        self.scope_stack.append(qualified_name)
        self.generic_visit(node)
        self.scope_stack.pop()
    
    def visit_Call(self, node: ast.Call) -> None:
        current = self.relationships[self.current_scope_name]
        if isinstance(node.func, ast.Name):
            current.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            current.calls.add(ast.unparse(node.func))
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        current = self.relationships[self.current_scope_name]
        if isinstance(node.ctx, ast.Load):
            current.reads_from.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            current.writes_to.add(node.id)
        self.generic_visit(node)
    
    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc:
            current = self.relationships[self.current_scope_name]
            if isinstance(node.exc, ast.Name):
                current.raises.add(node.exc.id)
            elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                current.raises.add(node.exc.func.id)
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            current = self.relationships[self.current_scope_name]
            if isinstance(node.type, ast.Name):
                current.catches.add(node.type.id)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        current = self.relationships[self.current_scope_name]
        module = node.module or ''
        for alias in node.names:
            current.imports.add(f"{module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        current = self.relationships[self.current_scope_name]
        for alias in node.names:
            current.imports.add(alias.name)
        self.generic_visit(node)

def analyze_relationships(source_code: str) -> Dict[str, NodeRelationships]:
    """Analyze relationships between nodes in Python source code."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        if "f-string expression part cannot include a backslash" in str(e):
            source_code = source_code.replace("\\", "")
            tree = ast.parse(source_code)
        else:
            raise e
            
    visitor = RelationshipVisitor()
    visitor.visit(tree)
    return visitor.relationships