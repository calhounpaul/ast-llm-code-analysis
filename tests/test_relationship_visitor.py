import pytest
from code_analyzer.analyzers.relationship_visitor import analyze_relationships, NodeRelationships

def test_class_relationships():
    source_code = """
class A:
    pass

class B(A):
    pass
"""
    relationships = analyze_relationships(source_code)

    assert "A" in relationships
    assert "B" in relationships
    assert relationships["B"].inherits_from == {"A"}


def test_function_relationships():
    source_code = """
def foo():
    pass

def bar(x: int) -> str:
    return foo()
"""
    relationships = analyze_relationships(source_code)

    assert "foo" in relationships
    assert "bar" in relationships
    assert relationships["bar"].calls == {"foo"}
    assert relationships["bar"].type_hints == {"return": "str", "x": "int"}


def test_import_relationships():
    source_code = """
import os
from math import sqrt, pi
"""
    relationships = analyze_relationships(source_code)

    assert "<module>" in relationships
    assert relationships["<module>"].imports == {"os", "math.sqrt", "math.pi"}


def test_variable_reads_and_writes():
    source_code = """
x = 10
y = x + 5
"""
    relationships = analyze_relationships(source_code)

    assert "<module>" in relationships
    assert relationships["<module>"].writes_to == {"x", "y"}
    assert relationships["<module>"].reads_from == {"x"}


def test_raise_and_catch():
    source_code = """
try:
    raise ValueError("An error occurred")
except ValueError:
    pass
"""
    relationships = analyze_relationships(source_code)

    assert "<module>" in relationships
    assert relationships["<module>"].raises == {"ValueError"}
    assert relationships["<module>"].catches == {"ValueError"}


def test_nested_scopes():
    source_code = """
class MyClass:
    def method(self):
        def inner():
            return 42
"""
    relationships = analyze_relationships(source_code)

    assert "MyClass" in relationships
    assert "MyClass.method" in relationships
    assert "MyClass.method.inner" in relationships
    assert relationships["MyClass.method.inner"].defined_in == "MyClass.method"


def test_f_string_with_backslash():
    source_code = """
value = 42
f"Result: {value}"
"""
    relationships = analyze_relationships(source_code)

    assert "<module>" in relationships
    assert "value" in relationships["<module>"].writes_to


def test_merge_relationships():
    r1 = NodeRelationships(
        calls={"foo"},
        reads_from={"x"},
        writes_to={"y"}
    )
    r2 = NodeRelationships(
        calls={"bar"},
        reads_from={"z"}
    )
    r1.merge(r2)

    assert r1.calls == {"foo", "bar"}
    assert r1.reads_from == {"x", "z"}
    assert r1.writes_to == {"y"}


if __name__ == "__main__":
    pytest.main()
