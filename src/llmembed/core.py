from typing import Any, List, Optional, Union

from .backends.transformers_backend import TransformersBackend
from .interfaces import Backend

# Try importing VLLMBackend, might fail if vllm is not installed or dependencies missing
try:
    from .backends.vllm_backend import VLLMBackend
except ImportError:
    VLLMBackend = None # type: ignore

class Encoder:
    def __init__(
        self,
        model_name: str,
        backend: str = "transformers",
        device: Optional[str] = None,
        quantization: Optional[str] = None
    ):
        """
        Initialize the Encoder.

        Args:
            model_name: Model identifier.
            backend: Backend to use ('transformers', 'vllm').
            device: Device ('cpu', 'cuda', etc.). If None, auto-detects.
            quantization: Quantization config ('4bit', '8bit', or None).
        """
        self.backend_name = backend
        self.backend_instance: Backend
        
        if backend == "transformers":
            self.backend_instance = TransformersBackend(model_name, device, quantization)
        elif backend == "vllm":
            # Check if VLLMBackend class is available
            if VLLMBackend is None:
                # Try importing again to see specific error or if it was just skipped
                try:
                    from .backends.vllm_backend import VLLMBackend as VBackend
                    self.backend_instance = VBackend(model_name, device, quantization)
                except ImportError as e:
                    raise ImportError(f"VLLM backend requires 'vllm' installed. Error: {e}")
            else:
                 self.backend_instance = VLLMBackend(model_name, device, quantization)
        else:
            raise ValueError(f"Unknown backend: {backend}")

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
