# ast-llm-code-analysis/code_analyzer/utils/model_singleton.py

from vllm import LLM

class ModelManager:
    """Singleton manager for LLM model instances."""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize_model(cls, model_name: str) -> None:
        """Initialize or reinitialize the model with the specified name."""
        cls._model = LLM(model=model_name, trust_remote_code=True)

    @classmethod
    def get_model(cls, model_name: str = None) -> LLM:
        """
        Get the model instance, initializing with the specified name if needed.
        
        Args:
            model_name: Optional model name to use if model needs initialization
        
        Returns:
            LLM: The model instance
        """
        if cls._model is None:
            if model_name is None:
                model_name = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
            cls.initialize_model(model_name)
        return cls._model

# Create singleton instance
model_manager = ModelManager()

# Expose key functions at module level
def initialize_model(model_name: str) -> None:
    """Initialize the global model instance."""
    model_manager.initialize_model(model_name)

def get_model(model_name: str = None) -> LLM:
    """Get the global model instance."""
    return model_manager.get_model(model_name)