import pytest
import ast
from pathlib import Path
from code_analyzer.analyzers.ast_analyzer import extract_map_of_nodes, ASTAnalyzer
from code_analyzer.models.data_classes import ModuleInfo, NodeRelationships
from unittest.mock import Mock

# Mock data for tests
MOCK_SOURCE = """
import os
import sys
from typing import List, Dict

class SampleClass:
    def __init__(self, x: int):
        self.x = x

    def method(self):
        pass

def sample_function(a: str) -> int:
    return len(a)
"""

MOCK_RELATIONSHIPS = {
    "SampleClass": NodeRelationships(defined_in="module", inherits_from=set(), calls=set(), contains=set(), imports=set()),
    "sample_function": NodeRelationships(defined_in="module", calls=set(), reads_from=set(), writes_to=set(), raises=set(), catches=set(), type_hints={})
}

MOCK_REPO_PATH = Path("/mock/repo")
MOCK_FILE_PATH = Path("/mock/repo/mock_file.py")
MOCK_TREE = "Mocked human-readable tree"

@pytest.fixture
def ast_analyzer():
    """Fixture for initializing ASTAnalyzer with mock data."""
    return ASTAnalyzer(
        source=MOCK_SOURCE,
        file_path=str(MOCK_FILE_PATH),
        relationships=MOCK_RELATIONSHIPS,
        repo_path=MOCK_REPO_PATH,
        human_readable_tree=MOCK_TREE,
        llm_manager=Mock()  # Mocked LLMQueryManager
    )

def test_minimal_extract_map_of_nodes_debug():
    """Minimal test for extract_map_of_nodes with debug output."""
    source = """
import os
def foo(): pass
class Bar: pass
"""
    result = extract_map_of_nodes(source)
    print("Result:", result)
    assert "FunctionDef" in result, f"FunctionDef missing: {result}"
    assert "ClassDef" in result, f"ClassDef missing: {result}"
    assert result["FunctionDef"] == ["foo"], f"Unexpected FunctionDef: {result['FunctionDef']}"
    assert result["ClassDef"] == ["Bar"], f"Unexpected ClassDef: {result['ClassDef']}"



def test_extract_map_of_nodes_basic():
    """Test extract_map_of_nodes function with valid input."""
    result = extract_map_of_nodes(MOCK_SOURCE)
    assert "FunctionDef" in result, f"FunctionDef key missing in result: {result}"
    assert "ClassDef" in result, f"ClassDef key missing in result: {result}"
    assert sorted(result["FunctionDef"]) == sorted(["__init__", "method", "sample_function"]), f"Unexpected FunctionDef result: {result['FunctionDef']}"
    assert sorted(result["ClassDef"]) == sorted(["SampleClass"]), f"Unexpected ClassDef result: {result['ClassDef']}"


def test_extract_map_of_nodes_with_escape_sequences():
    """Test extract_map_of_nodes handles backslash in f-strings."""
    source_with_escape = "f'{\\n}'"
    result = extract_map_of_nodes(source_with_escape)
    assert result == {}  # Expect no nodes in invalid source

def test_extract_map_of_nodes_with_escape_sequences():
    """Test extract_map_of_nodes handles backslash in f-strings."""
    source_with_escape = "f'{\\n}'"
    result = extract_map_of_nodes(source_with_escape)
    assert result == {}

def test_ast_analyzer_initialization(ast_analyzer):
    """Test that ASTAnalyzer initializes correctly."""
    assert ast_analyzer.source == MOCK_SOURCE
    assert ast_analyzer.file_path == str(MOCK_FILE_PATH)
    assert ast_analyzer.repo_path == MOCK_REPO_PATH
    assert ast_analyzer.human_readable_tree == MOCK_TREE
    assert isinstance(ast_analyzer.module, ModuleInfo)

def test_ast_analyzer_visit_module(ast_analyzer):
    """Test visit_Module populates module information."""
    module_node = ast.parse(MOCK_SOURCE)
    ast_analyzer.visit_Module(module_node)
    assert ast_analyzer.module.docstring is None
    assert "SampleClass" in ast_analyzer.module.classes
    assert "sample_function" in ast_analyzer.module.functions

def test_ast_analyzer_visit_classdef(ast_analyzer):
    """Test visit_ClassDef processes class definitions."""
    class_node = [node for node in ast.walk(ast.parse(MOCK_SOURCE)) if isinstance(node, ast.ClassDef)][0]
    ast_analyzer.visit_ClassDef(class_node)
    assert class_node.name in MOCK_RELATIONSHIPS
    assert class_node.name in ast_analyzer.module.classes

def test_ast_analyzer_visit_functiondef(ast_analyzer):
    """Test visit_FunctionDef processes function definitions."""
    func_node = [node for node in ast.walk(ast.parse(MOCK_SOURCE)) if isinstance(node, ast.FunctionDef)][0]
    ast_analyzer.visit_FunctionDef(func_node)
    assert func_node.name in MOCK_RELATIONSHIPS
    assert func_node.name in ast_analyzer.module.functions

def test_ast_analyzer_analyze(ast_analyzer):
    """Test the full analysis workflow."""
    module_info = ast_analyzer.analyze()
    assert isinstance(module_info, ModuleInfo)
    assert "SampleClass" in module_info.classes
    assert "sample_function" in module_info.functions
