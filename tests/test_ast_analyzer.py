#ast-llm-code-analysis/tests/test_ast_analyzer.py

import pytest
from pathlib import Path
import ast
from code_analyzer.analyzers.ast_analyzer import ASTAnalyzer
from code_analyzer.models.data_classes import ClassInfo, FunctionInfo, ModuleInfo
from typing import Dict

# Test fixtures
@pytest.fixture
def sample_source_code():
    return '''
class Person:
    """A simple person class."""
    def __init__(self, name: str):
        """Initialize person with name."""
        self.name = name
        
    @property
    def greeting(self) -> str:
        """Return greeting."""
        return f"Hello, {self.name}!"
        
def calculate_sum(a: int, b: int = 0) -> int:
    """Calculate sum of two numbers."""
    return a + b

CONSTANT = 42
'''

@pytest.fixture
def mock_relationships():
    return {
        'Person': {
            'calls': {'__init__'},
            'inherits_from': set(),
            'reads_from': {'name'},
            'writes_to': {'name'},
            'raises': set(),
            'catches': set(),
            'type_hints': {'name': 'str'},
            'defined_in': '<module>'
        },
        '__init__': {
            'calls': set(),
            'reads_from': {'name'},
            'writes_to': {'self.name'},
            'raises': set(),
            'catches': set(),
            'type_hints': {'name': 'str'},
            'defined_in': 'Person'
        },
        'greeting': {
            'calls': set(),
            'reads_from': {'self.name'},
            'writes_to': set(),
            'raises': set(),
            'catches': set(),
            'type_hints': {'return': 'str'},
            'defined_in': 'Person'
        },
        'calculate_sum': {
            'calls': set(),
            'reads_from': {'a', 'b'},
            'writes_to': set(),
            'raises': set(),
            'catches': set(),
            'type_hints': {'a': 'int', 'b': 'int', 'return': 'int'},
            'defined_in': '<module>'
        }
    }

@pytest.fixture
def analyzer(sample_source_code, mock_relationships, tmp_path):
    file_path = tmp_path / "test.py"
    repo_path = tmp_path
    human_readable_tree = f"/base/{file_path.relative_to(tmp_path)}"
    return ASTAnalyzer(
        source=sample_source_code,
        file_path=str(file_path),
        relationships=mock_relationships,
        repo_path=repo_path,
        human_readable_tree=human_readable_tree
    )

# Test cases
def test_module_analysis(analyzer):
    """Test basic module analysis."""
    module = analyzer.visit(ast.parse(analyzer.source))
    
    assert isinstance(analyzer.module, ModuleInfo)
    assert len(analyzer.module.classes) == 1
    assert len(analyzer.module.functions) == 1
    assert len(analyzer.module.constants) == 1
    assert analyzer.module.constants['CONSTANT'] == '42'

def test_class_analysis(analyzer):
    """Test class analysis capabilities."""
    module = analyzer.visit(ast.parse(analyzer.source))
    person_class = analyzer.module.classes['Person']
    
    assert isinstance(person_class, ClassInfo)
    assert person_class.name == 'Person'
    assert person_class.docstring == 'A simple person class.'
    assert len(person_class.methods) == 2  # __init__ and greeting
    assert 'greeting' in person_class.methods
    assert person_class.methods['greeting'].is_property

def test_function_analysis(analyzer):
    """Test function analysis capabilities."""
    module = analyzer.visit(ast.parse(analyzer.source))
    calc_sum = analyzer.module.functions['calculate_sum']
    
    assert isinstance(calc_sum, FunctionInfo)
    assert calc_sum.name == 'calculate_sum'
    assert calc_sum.docstring == 'Calculate sum of two numbers.'
    assert len(calc_sum.parameters) == 2
    assert calc_sum.parameters[0].name == 'a'
    assert calc_sum.parameters[0].annotation == 'int'
    assert calc_sum.parameters[1].name == 'b'
    assert calc_sum.parameters[1].default_value == '0'
    assert calc_sum.return_annotation == 'int'

def test_method_analysis(analyzer):
    """Test method analysis capabilities."""
    module = analyzer.visit(ast.parse(analyzer.source))
    init_method = analyzer.module.classes['Person'].methods['__init__']
    
    assert isinstance(init_method, FunctionInfo)
    assert init_method.name == '__init__'
    assert init_method.is_method
    assert len(init_method.parameters) == 2  # self and name
    assert init_method.parameters[1].annotation == 'str'

def test_property_analysis(analyzer):
    """Test property method analysis."""
    module = analyzer.visit(ast.parse(analyzer.source))
    greeting = analyzer.module.classes['Person'].methods['greeting']
    
    assert isinstance(greeting, FunctionInfo)
    assert greeting.is_property
    assert greeting.return_annotation == 'str'
    assert not greeting.parameters  # Properties don't show self parameter

def test_relationship_tracking(analyzer):
    """Test relationship tracking in analysis."""
    module = analyzer.visit(ast.parse(analyzer.source))
    init_method = analyzer.module.classes['Person'].methods['__init__']
    
    # Check writes tracking
    assert 'self.name' in init_method.assigns_to
    
    # Check reads tracking
    assert 'name' in init_method.reads_from

def test_source_extraction(analyzer):
    """Test source code extraction capabilities."""
    module = analyzer.visit(ast.parse(analyzer.source))
    calc_sum = analyzer.module.functions['calculate_sum']
    
    assert 'def calculate_sum' in calc_sum.raw_code
    assert 'return a + b' in calc_sum.raw_code

def test_prompt_generation(analyzer):
    """Test prompt template generation."""
    module = analyzer.visit(ast.parse(analyzer.source))
    calc_sum = analyzer.module.functions['calculate_sum']
    
    assert isinstance(calc_sum.prompt, str)
    assert 'analyze this Python function' in calc_sum.prompt
    assert 'calculate_sum' in calc_sum.prompt
    assert 'Function Context' in calc_sum.prompt

def test_error_handling(analyzer):
    """Test handling of syntax errors."""
    invalid_source = """
    def invalid_function(
        print("Missing closing parenthesis"
    """
    with pytest.raises(SyntaxError):
        analyzer.visit(ast.parse(invalid_source))

def test_type_hint_extraction(analyzer):
    """Test extraction of type hints."""
    module = analyzer.visit(ast.parse(analyzer.source))
    calc_sum = analyzer.module.functions['calculate_sum']
    
    assert calc_sum.parameters[0].annotation == 'int'
    assert calc_sum.parameters[1].annotation == 'int'
    assert calc_sum.return_annotation == 'int'

def test_decorator_handling(sample_source_code, mock_relationships, tmp_path):
    """Test handling of decorators."""
    source = '''
    class Calculator:
        @staticmethod
        def add(a: int, b: int) -> int:
            return a + b
            
        @classmethod
        def create(cls) -> 'Calculator':
            return cls()
    '''
    
    analyzer = ASTAnalyzer(
        source=source,
        file_path=str(tmp_path / "test.py"),
        relationships=mock_relationships,
        repo_path=tmp_path,
        human_readable_tree="/base/test.py"
    )
    
    module = analyzer.visit(ast.parse(source))
    calculator = analyzer.module.classes['Calculator']
    
    assert calculator.methods['add'].is_static
    assert calculator.methods['create'].is_class_method

if __name__ == '__main__':
    pytest.main([__file__])