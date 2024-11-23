#ast-llm-code-analysis/tests/test_relationship_visitor.py

import pytest
from typing import Dict
import ast
from code_analyzer.analyzers.relationship_visitor import RelationshipVisitor, NodeRelationships

def analyze_code(code: str) -> Dict[str, NodeRelationships]:
    """Helper function to analyze Python code and return relationships."""
    tree = ast.parse(code)
    visitor = RelationshipVisitor()
    visitor.visit(tree)
    return visitor.relationships

def test_class_inheritance():
    """Test that class inheritance relationships are correctly detected."""
    code = """
    class Parent:
        pass
        
    class Child(Parent):
        pass
        
    class MultiChild(Parent, OtherParent):
        pass
    """
    
    relationships = analyze_code(code)
    
    assert "Child" in relationships
    assert "Parent" in relationships["Child"].inherits_from
    
    assert "MultiChild" in relationships
    assert "Parent" in relationships["MultiChild"].inherits_from
    assert "OtherParent" in relationships["MultiChild"].inherits_from

def test_function_calls():
    """Test that function calls are correctly detected."""
    code = """
    def helper():
        pass
        
    def main():
        helper()
        other_func()
        obj.method()
    """
    
    relationships = analyze_code(code)
    
    assert "main" in relationships
    assert "helper" in relationships["main"].calls
    assert "other_func" in relationships["main"].calls
    assert "obj.method" in relationships["main"].calls

def test_method_decorators():
    """Test that method decorators are correctly analyzed."""
    code = """
    class MyClass:
        @property
        def prop(self):
            pass
            
        @staticmethod
        def static_method():
            pass
            
        @classmethod
        def class_method(cls):
            pass
    """
    
    relationships = analyze_code(code)
    
    assert "prop" in relationships
    assert "property" in relationships["prop"].calls
    
    assert "static_method" in relationships
    assert "staticmethod" in relationships["static_method"].calls
    
    assert "class_method" in relationships
    assert "classmethod" in relationships["class_method"].calls

def test_variable_access():
    """Test that variable reads and writes are correctly tracked."""
    code = """
    def process_data():
        x = 1  # write
        y = x + z  # read x and z
        return y  # read y
    """
    
    relationships = analyze_code(code)
    
    assert "process_data" in relationships
    assert "x" in relationships["process_data"].writes_to
    assert "x" in relationships["process_data"].reads_from
    assert "y" in relationships["process_data"].writes_to
    assert "y" in relationships["process_data"].reads_from
    assert "z" in relationships["process_data"].reads_from

def test_import_tracking():
    """Test that imports are correctly tracked."""
    code = """
    import os
    from pathlib import Path
    from datetime import datetime as dt
    
    def use_imports():
        os.path.join()
        Path('.')
        dt.now()
    """
    
    relationships = analyze_code(code)
    
    assert "use_imports" in relationships
    assert "os" in relationships["use_imports"].imports
    assert "pathlib.Path" in relationships["use_imports"].imports
    assert "datetime.datetime" in relationships["use_imports"].imports

def test_exception_handling():
    """Test that raised and caught exceptions are tracked."""
    code = """
    def risky_operation():
        try:
            raise ValueError("error")
        except (TypeError, ValueError) as e:
            raise RuntimeError from e
    """
    
    relationships = analyze_code(code)
    
    assert "risky_operation" in relationships
    assert "ValueError" in relationships["risky_operation"].raises
    assert "RuntimeError" in relationships["risky_operation"].raises
    assert "TypeError" in relationships["risky_operation"].catches
    assert "ValueError" in relationships["risky_operation"].catches

def test_type_hints():
    """Test that type hints are correctly extracted."""
    code = """
    from typing import List, Optional
    
    def process_items(items: List[str], limit: Optional[int] = None) -> bool:
        return True
    """
    
    relationships = analyze_code(code)
    
    assert "process_items" in relationships
    assert relationships["process_items"].type_hints["items"] == "List[str]"
    assert relationships["process_items"].type_hints["limit"] == "Optional[int]"
    assert relationships["process_items"].type_hints["return"] == "bool"

def test_nested_scope():
    """Test that nested scope relationships are correctly tracked."""
    code = """
    class Outer:
        class Inner:
            def inner_method(self):
                pass
                
        def outer_method(self):
            def nested_func():
                pass
            nested_func()
    """
    
    relationships = analyze_code(code)
    
    assert "Inner" in relationships
    assert relationships["Inner"].defined_in == "Outer"
    
    assert "inner_method" in relationships
    assert relationships["inner_method"].defined_in == "Outer.Inner"
    
    assert "outer_method" in relationships
    assert relationships["outer_method"].defined_in == "Outer"
    assert "nested_func" in relationships["outer_method"].calls

def test_complex_method_chain():
    """Test that complex method chains are correctly analyzed."""
    code = """
    class DataProcessor:
        def process(self):
            self.validate()
            data = self.fetch_data()
            self.transform(data)
            return self.save_results()
    """
    
    relationships = analyze_code(code)
    
    assert "process" in relationships
    assert "self.validate" in relationships["process"].calls
    assert "self.fetch_data" in relationships["process"].calls
    assert "self.transform" in relationships["process"].calls
    assert "self.save_results" in relationships["process"].calls

def test_attribute_access():
    """Test that attribute access is correctly tracked."""
    code = """
    class Config:
        def update_settings(self):
            self.host = "localhost"
            port = self.port
            self.timeout = self.default_timeout
    """
    
    relationships = analyze_code(code)
    
    assert "update_settings" in relationships
    assert "host" in relationships["update_settings"].writes_to
    assert "port" in relationships["update_settings"].reads_from
    assert "timeout" in relationships["update_settings"].writes_to
    assert "default_timeout" in relationships["update_settings"].reads_from

def test_relationships_merge():
    """Test that NodeRelationships can be correctly merged."""
    rel1 = NodeRelationships()
    rel1.calls.add("func1")
    rel1.imports.add("module1")
    rel1.reads_from.add("var1")
    
    rel2 = NodeRelationships()
    rel2.calls.add("func2")
    rel2.imports.add("module2")
    rel2.writes_to.add("var2")
    
    rel1.merge(rel2)
    
    assert "func1" in rel1.calls
    assert "func2" in rel1.calls
    assert "module1" in rel1.imports
    assert "module2" in rel1.imports
    assert "var1" in rel1.reads_from
    assert "var2" in rel1.writes_to