#ast-llm-code-analysis/code_analyzer/models/data_classes.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from collections import defaultdict

@dataclass
class Parameter:
    """Represents a function or method parameter."""
    name: str
    annotation: Optional[str] = None
    default_value: Optional[str] = None
    
    def to_compact(self) -> Dict[str, Any]:
        """Convert to a compact dictionary representation."""
        output = CompactOutput()
        output.add('name', self.name)
        output.add('annotation', self.annotation)
        output.add('default_value', self.default_value)
        return output.to_dict()

@dataclass
class AnalysisResults:
    """Stores the results of LLM analysis."""
    initial_analysis: Optional[str] = None
    summary: Optional[str] = None
    
    def to_compact(self) -> Dict[str, Any]:
        """Convert to a compact dictionary representation."""
        output = CompactOutput()
        output.add('initial_analysis', self.initial_analysis)
        output.add('summary', self.summary)
        return output.to_dict()

@dataclass
class FunctionInfo:
    """Stores information about a function or method."""
    name: str
    parameters: List[Parameter]
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_method: bool = False
    is_static: bool = False
    is_class_method: bool = False
    is_property: bool = False
    calls: List[str] = field(default_factory=list)
    assigns_to: List[str] = field(default_factory=list)
    reads_from: List[str] = field(default_factory=list)
    raw_code: str = ""
    prompt: str = ""
    llm_analysis: Optional[AnalysisResults] = None
    
    def to_compact(self) -> Dict[str, Any]:
        """Convert to a compact dictionary representation."""
        output = CompactOutput()
        output.add('name', self.name)
        output.add('parameters', [p.to_compact() for p in self.parameters])
        output.add('return_annotation', self.return_annotation)
        output.add('decorators', self.decorators)
        output.add('docstring', self.docstring)
        output.add('raw_code', self.raw_code)
        output.add('prompt', self.prompt)
        
        if self.llm_analysis:
            output.add('llm_analysis', self.llm_analysis.to_compact())
        
        # Only include boolean flags that are True
        for flag in ['is_method', 'is_static', 'is_class_method', 'is_property']:
            if getattr(self, flag):
                output.add(flag, True)
        
        output.add('calls', self.calls)
        output.add('assigns_to', self.assigns_to)
        output.add('reads_from', self.reads_from)
        return output.to_dict()

@dataclass
class ClassInfo:
    """Stores information about a class."""
    name: str
    bases: List[str]
    methods: Dict[str, FunctionInfo] = field(default_factory=dict)
    properties: List[str] = field(default_factory=list)
    class_variables: Dict[str, str] = field(default_factory=dict)
    instance_variables: Set[str] = field(default_factory=set)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    raw_code: str = ""
    prompt: str = ""
    llm_analysis: Optional[AnalysisResults] = None
    
    def to_compact(self) -> Dict[str, Any]:
        """Convert to a compact dictionary representation."""
        output = CompactOutput()
        output.add('name', self.name)
        output.add('bases', self.bases)
        output.add('raw_code', self.raw_code)
        output.add('prompt', self.prompt)
        
        if self.llm_analysis:
            output.add('llm_analysis', self.llm_analysis.to_compact())
        
        if self.methods:
            output.add('methods', {
                name: method.to_compact() 
                for name, method in self.methods.items()
            })
        
        output.add('properties', self.properties)
        output.add('class_variables', self.class_variables)
        output.add('instance_variables', sorted(list(self.instance_variables)))
        output.add('decorators', self.decorators)
        output.add('docstring', self.docstring)
        return output.to_dict()

@dataclass
class ModuleInfo:
    """Stores information about a Python module."""
    path: str
    imports: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    globals: Dict[str, Any] = field(default_factory=dict)
    class_instantiations: List[Dict[str, Union[str, List[str]]]] = field(default_factory=list)
    raw_code: str = ""
    prompt: str = ""
    error: Optional[str] = None
    llm_analysis: Optional[AnalysisResults] = None
    
    def to_compact(self) -> Dict[str, Any]:
        """Convert to a compact dictionary representation."""
        output = CompactOutput()
        output.add('path', self.path)
        output.add('raw_code', self.raw_code)
        output.add('prompt', self.prompt)
        
        if self.llm_analysis:
            output.add('llm_analysis', self.llm_analysis.to_compact())
        
        # Only include non-empty import types
        imports = {k: v for k, v in self.imports.items() if v}
        if imports:
            output.add('imports', imports)
        
        if self.classes:
            output.add('classes', {
                name: cls.to_compact() 
                for name, cls in self.classes.items()
            })
        
        if self.functions:
            output.add('functions', {
                name: func.to_compact() 
                for name, func in self.functions.items()
            })
        
        if self.error:
            output.add('error', self.error)
            
        output.add('constants', self.constants)
        output.add('globals', self.globals)
        output.add('class_instantiations', self.class_instantiations)
        return output.to_dict()

@dataclass
class CompactOutput:
    """Helper class to manage compact output format."""
    data: Dict[str, Any] = field(default_factory=dict)
    empty_keys: Set[str] = field(default_factory=set)
    
    def add(self, key: str, value: Any) -> None:
        """Add a key-value pair to the output, tracking empty collections."""
        if value:
            if isinstance(value, dict) and not any(value.values()):
                self.empty_keys.add(key)
            elif isinstance(value, (list, set)) and not value:
                self.empty_keys.add(key)
            else:
                self.data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to final dictionary format, including empty keys if any."""
        if self.empty_keys:
            self.data['empty_keys'] = sorted(list(self.empty_keys))
        return self.data

@dataclass
class NodeRelationships:
    """Tracks relationships between AST nodes."""
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