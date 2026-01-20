from typing import Any, List, Optional, Union
import logging

from .backends.transformers_backend import TransformersBackend
from .interfaces import Backend

try:
    from .backends.vllm_backend import VLLMBackend
except ImportError:
    VLLMBackend = None

logger = logging.getLogger(__name__)

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
            backend: Backend to use ('transformers' or 'vllm').
            device: Device ('cpu', 'cuda', etc.). If None, auto-detects.
            quantization: Quantization config ('4bit', '8bit', or None for transformers; 
                          'fp8', 'awq', 'gptq' etc. for vllm).
            **kwargs: Additional arguments passed to the backend.
                      For vllm, this includes 'tensor_parallel_size', 'gpu_memory_utilization', etc.
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
        elif backend == "vllm":
            if VLLMBackend is None:
                raise ImportError(
                    "The 'vllm' backend is not available. "
                    "Please install `vllm` and ensure `.backends.vllm_backend` exists."
                )
            
            self.backend_instance = VLLMBackend(
                model_name,
                device=device,
                quantization=quantization,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported backends are 'transformers' and 'vllm'.")

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
                        Note: vLLM backend typically only supports the last layer (-1).
            **kwargs: Backend specific arguments.

        Returns:
            Embeddings as numpy array or torch tensor.
        """
        return self.backend_instance.encode(
            text, pooling=pooling, layer_index=layer_index, **kwargs
        )