#ast-llm-code-analysis/tests/test_repo_analyzer.py

import unittest
from pathlib import Path
import tempfile
import shutil
import json
import os
from unittest.mock import patch, MagicMock, ANY

from code_analyzer.analyzers.repo_analyzer import RepoAnalyzer
from code_analyzer.models.data_classes import ModuleInfo, ClassInfo, FunctionInfo, Parameter, AnalysisResults

class TestRepoAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up temporary directory and sample files for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_files = {
            'simple.py': '''
def greet(name: str) -> str:
    """Simple greeting function"""
    return f"Hello, {name}!"
''',
            'complex.py': '''
from typing import List, Optional

class Person:
    """Represents a person"""
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def greet(self) -> str:
        return f"Hi, I'm {self.name}!"

def process_people(people: List[Person]) -> List[str]:
    """Process a list of people"""
    return [person.greet() for person in people]

GREETING = "Welcome!"
'''
        }
        
        # Create sample files in temp directory
        for filename, content in self.sample_files.items():
            file_path = Path(self.temp_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Initialize analyzer with temp directory
        self.analyzer = RepoAnalyzer(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory after tests"""
        shutil.rmtree(self.temp_dir)

    @patch('code_analyzer.analyzers.repo_analyzer.query_qwen')
    def test_analyze_single_file(self, mock_query):
        """Test analysis of a single Python file"""
        # Mock LLM responses
        mock_query.return_value = "Mock analysis"
        
        # Analyze simple.py
        file_path = str(Path(self.temp_dir) / 'simple.py')
        module_info = self.analyzer.analyze_file(file_path)
        
        # Verify basic module properties
        self.assertIsInstance(module_info, ModuleInfo)
        self.assertEqual(module_info.path, file_path)
        self.assertIn('greet', module_info.functions)
        
        # Verify function analysis
        greet_func = module_info.functions['greet']
        self.assertEqual(greet_func.name, 'greet')
        self.assertEqual(greet_func.return_annotation, 'str')
        self.assertEqual(len(greet_func.parameters), 1)
        self.assertEqual(greet_func.parameters[0].name, 'name')
        self.assertEqual(greet_func.parameters[0].annotation, 'str')
        
        # Verify LLM was called appropriate number of times
        expected_calls = 2  # One for module, one for function
        self.assertEqual(mock_query.call_count, expected_calls)

    @patch('code_analyzer.analyzers.repo_analyzer.query_qwen')
    def test_analyze_complex_file(self, mock_query):
        """Test analysis of a file with multiple components"""
        # Mock LLM responses
        mock_query.return_value = "Mock analysis"
        
        # Analyze complex.py
        file_path = str(Path(self.temp_dir) / 'complex.py')
        module_info = self.analyzer.analyze_file(file_path)
        
        # Verify imports
        self.assertIn('typing', str(module_info.imports))
        
        # Verify class analysis
        self.assertIn('Person', module_info.classes)
        person_class = module_info.classes['Person']
        self.assertEqual(len(person_class.methods), 2)  # __init__ and greet
        
        # Verify method analysis
        greet_method = person_class.methods['greet']
        self.assertTrue(greet_method.is_method)
        self.assertEqual(greet_method.return_annotation, 'str')
        
        # Verify function analysis
        self.assertIn('process_people', module_info.functions)
        process_func = module_info.functions['process_people']
        self.assertEqual(process_func.return_annotation, 'List[str]')
        
        # Verify constants
        self.assertIn('GREETING', module_info.constants)
        self.assertEqual(module_info.constants['GREETING'], '"Welcome!"')

    def test_analyze_directory(self):
        """Test analysis of entire directory"""
        modules = self.analyzer.analyze_directory()
        
        # Verify all files were analyzed
        expected_files = {str(Path(self.temp_dir) / f) for f in self.sample_files.keys()}
        analyzed_files = set(modules.keys())
        self.assertEqual(expected_files, analyzed_files)
        
        # Verify each module was analyzed properly
        for module in modules.values():
            self.assertIsInstance(module, ModuleInfo)
            self.assertIsNotNone(module.raw_code)

    def test_save_analysis(self):
        """Test saving analysis results to JSON"""
        # Analyze directory
        self.analyzer.analyze_directory()
        
        # Save analysis
        output_path = str(Path(self.temp_dir) / 'analysis.json')
        self.analyzer.save_analysis(output_path)
        
        # Verify JSON file exists and is valid
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check basic structure
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), len(self.sample_files))
        
        # Verify each module's data
        for module_path, module_data in data.items():
            self.assertIn('path', module_data)
            self.assertIn('raw_code', module_data)

    def test_get_all_functions(self):
        """Test retrieving all functions from analyzed files"""
        self.analyzer.analyze_directory()
        functions = self.analyzer.get_all_functions()
        
        # Verify function count (greet, __init__, greet method, process_people)
        self.assertEqual(len(functions), 4)
        
        # Verify function types
        function_names = {f.name for f in functions}
        self.assertIn('greet', function_names)
        self.assertIn('process_people', function_names)
        self.assertIn('__init__', function_names)

    def test_get_all_classes(self):
        """Test retrieving all classes from analyzed files"""
        self.analyzer.analyze_directory()
        classes = self.analyzer.get_all_classes()
        
        # Verify class count
        self.assertEqual(len(classes), 1)
        
        # Verify class properties
        person_class = classes[0]
        self.assertEqual(person_class.name, 'Person')
        self.assertEqual(len(person_class.methods), 2)

    def test_get_dependency_graph(self):
        """Test generating dependency graph from imports"""
        self.analyzer.analyze_directory()
        graph = self.analyzer.get_dependency_graph()
        
        # Verify graph structure
        self.assertIsInstance(graph, dict)
        self.assertEqual(len(graph), len(self.sample_files))
        
        # Check dependencies
        complex_path = str(Path(self.temp_dir) / 'complex.py')
        self.assertIn(complex_path, graph)
        
        # simple.py should have no dependencies
        simple_path = str(Path(self.temp_dir) / 'simple.py')
        self.assertEqual(len(graph[simple_path]), 0)

    @patch('code_analyzer.analyzers.repo_analyzer.query_qwen')
    def test_error_handling(self, mock_query):
        """Test handling of syntax errors in Python files"""
        # Create file with syntax error
        error_file = Path(self.temp_dir) / 'error.py'
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write('def invalid_syntax(:')  # Missing parameter name
        
        # Analyze file
        module_info = self.analyzer.analyze_file(str(error_file))
        
        # Verify error was captured
        self.assertIsNotNone(module_info.error)
        self.assertIn('SyntaxError', module_info.error)
        
        # Verify LLM wasn't called for invalid file
        mock_query.assert_not_called()

if __name__ == '__main__':
    unittest.main()