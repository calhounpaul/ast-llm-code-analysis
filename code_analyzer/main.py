import argparse
import sys
from pathlib import Path
import logging
import os

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Python code repositories using LLM-powered code understanding."
    )
    
    parser.add_argument(
        "repo_path",
        type=str,
        help="Path to the repository or directory to analyze"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="analysis.json",
        help="Path to save the analysis output (default: analysis.json)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        help="Model name/path to use for analysis"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of LLM responses"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output of LLM queries and responses"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export raw LLM queries and responses to text files"
    )
    
    return parser.parse_args()


def get_app_dirs() -> tuple[Path, Path, Path, Path]:
    """Get application directories for cache, logs, LLM output, and debug."""
    cache_dir = Path(os.getenv('CODE_ANALYZER_CACHE_DIR', Path.home() / '.local' / 'share' / 'code_analyzer' / 'cache'))
    log_dir = Path(os.getenv('CODE_ANALYZER_LOG_DIR', Path.home() / '.local' / 'share' / 'code_analyzer' / 'logs'))
    llm_dir = Path(os.getenv('CODE_ANALYZER_LLM_DIR', Path.home() / '.local' / 'share' / 'code_analyzer' / 'llm_out'))
    debug_dir = Path(os.getenv('CODE_ANALYZER_DEBUG_DIR', Path.home() / '.local' / 'share' / 'code_analyzer' / 'debug'))
    
    return cache_dir, log_dir, llm_dir, debug_dir

def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration for the application."""
    # Get all directories but only use log_dir for logging
    cache_dir, log_dir, llm_dir, debug_dir = get_app_dirs()
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'code_analyzer.log'
    
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
        
    # Log the paths being used
    logger = logging.getLogger(__name__)
    logger.debug(f"Log file location: {log_file}")

def main() -> int:
    args = parse_arguments()
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Get all directories
        cache_dir, log_dir, llm_dir, debug_dir = get_app_dirs()
        
        # Create all directories
        for dir_path in [cache_dir, log_dir, llm_dir, debug_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        db_path = cache_dir / 'query_cache.db'
        logger.debug(f"Using cache database: {db_path}")
        
        repo_path = Path(args.repo_path)
        logger.debug(f"Analyzing repository at: {repo_path}")
        logger.debug(f"Repository path exists: {repo_path.exists()}")
        logger.debug(f"Repository path is directory: {repo_path.is_dir()}")
        if repo_path.is_dir():
            logger.debug(f"Directory contents: {list(repo_path.iterdir())}")
        
        from code_analyzer.analyzers.repo_analyzer import RepoAnalyzer
        from code_analyzer.utils.llm_query import LLMQueryManager
        
        # Create LLMQueryManager with all necessary paths
        llm_manager = LLMQueryManager(
            verbose=args.verbose,
            use_cache=not args.no_cache,
            cache_db=str(db_path),
            debug_dir=str(debug_dir) if args.debug else None,
            llm_out_dir=str(llm_dir) if args.export else None,
            export=args.export
        )
        
        logger.info("Creating analyzer instance...")
        analyzer = RepoAnalyzer(
            repo_path=args.repo_path,
            llm_manager=llm_manager
        )
        
        # Analyze the repository
        logger.info("Starting directory analysis...")
        modules = analyzer.analyze_directory()
        
        # Save results
        logger.info(f"Saving analysis results to {args.output}")
        analyzer.save_analysis(args.output)
        
        # Print summary statistics
        num_files = len(modules)
        num_functions = len(analyzer.get_all_functions())
        num_classes = len(analyzer.get_all_classes())
        
        logger.info(f"""
Analysis complete:
- Files analyzed: {num_files}
- Functions found: {num_functions}
- Classes found: {num_classes}
        """.strip())
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Unexpected error during analysis")
        return 1

if __name__ == "__main__":
    sys.exit(main())