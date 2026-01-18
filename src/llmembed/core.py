from typing import Any, List, Optional, Union

from .backends.transformers_backend import TransformersBackend
from .interfaces import Backend

# Try importing VLLMBackend, might fail if vllm is not installed or dependencies missing
# vLLM support removed in favor of focusing on transformers for hidden state extraction.
# VLLMBackend = None

class Encoder:
    def __init__(
        self,
        model_name: str,
        backend: str = "transformers",
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the Encoder.

        Args:
            model_name: Model identifier.
            backend: Backend to use (only 'transformers' is supported).
            device: Device ('cpu', 'cuda', etc.). If None, auto-detects.
            quantization: Quantization config ('4bit', '8bit', or None).
            **kwargs: Additional arguments passed to the backend
                      (e.g., model_kwargs).
        """
        self.backend_name = backend
        self.backend_instance: Backend
        
        if backend == "transformers":
            self.backend_instance = TransformersBackend(
                model_name, 
                device=device, 
                quantization=quantization, 
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Only 'transformers' is supported.")

    def encode(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        layer_index: Optional[int] = None,
        **kwargs: Any
    ) -> Any:
        """
        Encode text into embeddings.

        Args:
            text: Input text or list of texts.
            pooling: Pooling strategy ('mean', 'last_token', 'eos_token', 'prompt_eol',
                                     'pcoteol', 'ke').
            layer_index: Layer index to extract embeddings from.
                        Defaults to -2 for 'pcoteol'/'ke', and -1 for others.
            **kwargs: Backend specific arguments.

        Returns:
            Embeddings as numpy array or torch tensor.
        """
        return self.backend_instance.encode(
            text, pooling=pooling, layer_index=layer_index, **kwargs
        )