#ast-llm-code-analysis/code_analyzer/utils/model_singleton.py
from vllm import LLM

model: LLM = None

def initialize_model(model_name: str) -> None:
    """Initialize the global model instance."""
    global model
    if model is None:
        model = LLM(model=model_name, trust_remote_code=True)

def get_model() -> LLM:
    """Get the global model instance."""
    global model
    if model is None:
        initialize_model("Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")
    return model