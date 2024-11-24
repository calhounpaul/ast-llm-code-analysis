import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from code_analyzer.analyzers.repo_analyzer import RepoAnalyzer
from code_analyzer.models.data_classes import ModuleInfo, FunctionInfo, ClassInfo, AnalysisResults

@pytest.fixture
def mock_llm_manager():
    """Fixture to create a mock LLMQueryManager."""
    llm_manager = MagicMock()
    llm_manager.query.return_value = "Mock analysis result"
    llm_manager.query_multi_turn.return_value = "Mock summary"
    return llm_manager

@pytest.fixture
def repo_analyzer(mock_llm_manager):
    """Fixture to initialize a RepoAnalyzer instance with a mock LLM manager."""
    return RepoAnalyzer(repo_path="mock_repo", llm_manager=mock_llm_manager)

@patch("builtins.open", new_callable=mock_open, read_data="def mock_function(): pass")
@patch("code_analyzer.analyzers.repo_analyzer.extract_human_readable_file_tree", return_value="Mock tree")
@patch("code_analyzer.analyzers.repo_analyzer.analyze_relationships", return_value={"mock_relationships": []})
@patch("code_analyzer.analyzers.repo_analyzer.ASTAnalyzer")
def test_analyze_file(mock_ast_analyzer, mock_analyze_relationships, mock_extract_tree, mock_open_file, repo_analyzer):
    """Test analyze_file method."""
    # Mock the ASTAnalyzer visit method and module object
    mock_ast_instance = mock_ast_analyzer.return_value
    mock_ast_instance.module = ModuleInfo(
        path="mock_file.py",
        raw_code="def mock_function(): pass",
        functions={
            "mock_function": FunctionInfo(
                name="mock_function",
                prompt="Mock prompt",
                parameters=[]
            )
        },
        classes={}
    )
    
    module_info = repo_analyzer.analyze_file("mock_file.py")
    
    # Assertions
    assert module_info.path == "mock_file.py"
    assert "mock_function" in module_info.functions
    repo_analyzer.llm_manager.query.assert_called()  # Ensure LLM was queried

from unittest.mock import patch, MagicMock
from pathlib import Path

@patch("code_analyzer.analyzers.repo_analyzer.Path.rglob", return_value=[Path("mock_file1.py"), Path("mock_file2.py")])
@patch("code_analyzer.analyzers.repo_analyzer.RepoAnalyzer.analyze_file")
def test_analyze_directory(mock_analyze_file, mock_rglob, repo_analyzer):
    """Test analyze_directory method with mocked Path.is_file."""
    # Mock analyze_file to populate modules
    def mock_analyze_file_side_effect(file_path):
        print("Mock analyze_file called with:", file_path)
        module = ModuleInfo(path=file_path, raw_code="", functions={}, classes={})
        repo_analyzer.modules[file_path] = module
        return module

    mock_analyze_file.side_effect = mock_analyze_file_side_effect

    # Patch is_file to always return True
    with patch.object(Path, "is_file", return_value=True):
        # Run the method
        repo_analyzer.analyze_directory()

    # Debug: Verify the output from rglob and analyze_file
    print("Paths returned by rglob:", list(mock_rglob.return_value))
    print("Mock analyze_file calls:", mock_analyze_file.call_args_list)

    # Ensure analyze_file was called for each file
    mock_analyze_file.assert_any_call("mock_file1.py")
    mock_analyze_file.assert_any_call("mock_file2.py")

    # Assertions
    assert len(repo_analyzer.modules) == 2
    assert "mock_file1.py" in repo_analyzer.modules
    assert "mock_file2.py" in repo_analyzer.modules

def test_get_all_functions(repo_analyzer):
    """Test get_all_functions method."""
    repo_analyzer.modules = {
        "file1.py": ModuleInfo(
            path="file1.py",
            raw_code="",
            functions={
                "func1": FunctionInfo(name="func1", prompt="", parameters=[]),
                "func2": FunctionInfo(name="func2", prompt="", parameters=[])
            },
            classes={}
        )
    }
    
    functions = repo_analyzer.get_all_functions()
    assert len(functions) == 2
    assert functions[0].name == "func1"

def test_get_module_statistics(repo_analyzer):
    """Test get_module_statistics method."""
    repo_analyzer.modules = {
        "file1.py": ModuleInfo(
            path="file1.py",
            raw_code="def func1(): pass\nclass Class1:\n    def method1(self): pass",
            functions={"func1": FunctionInfo(name="func1", prompt="", parameters=[])},
            classes={
                "Class1": ClassInfo(
                    name="Class1",
                    bases=["object"],
                    methods={
                        "method1": FunctionInfo(name="method1", prompt="", parameters=[])
                    }
                )
            }
        )
    }
    
    stats = repo_analyzer.get_module_statistics()
    
    # Assertions
    assert stats["total_modules"] == 1
    assert stats["total_classes"] == 1
    assert stats["total_functions"] == 1
    assert stats["total_methods"] == 1
    assert stats["total_lines"] == 3

@patch("code_analyzer.analyzers.repo_analyzer.json.dump")
def test_save_analysis(mock_json_dump, repo_analyzer):
    """Test save_analysis method."""
    repo_analyzer.modules = {
        "file1.py": ModuleInfo(path="file1.py", raw_code="", functions={}, classes={})
    }
    
    repo_analyzer.save_analysis("mock_output.json")
    
    # Assertions
    mock_json_dump.assert_called_once()
