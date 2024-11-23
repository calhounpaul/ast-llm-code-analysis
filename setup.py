from setuptools import setup, find_packages

setup(
    name="code_analyzer",
    version="1.0.0",
    description="A tool for analyzing Python codebases using AST analysis and LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Code Analyzer Team",
    author_email="team@codeanalyzer.org",
    url="https://github.com/org/code-analyzer",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    package_data={
        "code_analyzer": [
            "assets/*.json",
        ],
    },
    install_requires=[
        "torch>=2.0.0",
        "vllm>=0.2.0",
        "transformers>=4.36.0",
        "pathlib>=1.0.1",
        "sqlite3-api>=2.0.1",
        "typing-extensions>=4.0.0",
        "dataclasses>=0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "flake8>=4.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "code-analyzer=code_analyzer.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)