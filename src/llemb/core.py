import logging
from typing import Any, List, Optional, Union

import torch
from tqdm import tqdm

from .backends.transformers_backend import TransformersBackend
from .interfaces import Backend

try:
    from .backends.vllm_backend import VLLMBackend
except ImportError:
    VLLMBackend = None  # type: ignore

logger = logging.getLogger(__name__)


class Encoder:
    def __init__(
        self,
        model_name: str,
        backend: str = "transformers",
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs: Any,
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
                model_name, device=device, quantization=quantization, **kwargs
            )
        elif backend == "vllm":
            if VLLMBackend is None:
                raise ImportError(
                    "The 'vllm' backend is not available. "
                    "Please install `vllm` and ensure `.backends.vllm_backend` exists."
                )

            # vLLM backend requires a strict string for device (e.g. "cuda").
            # If 'device' is None (auto), default to "cuda".
            vllm_device = device if device is not None else "cuda"

            self.backend_instance = VLLMBackend(
                model_name, device=vllm_device, quantization=quantization, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported backends are 'transformers' and 'vllm'."
            )

    def encode(
        self,
        text: Union[str, List[str]],
        pooling_method: Optional[str] = None,
        layer_index: Optional[int] = None,
        prompt_template: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Encode text into embeddings.

        Args:
            text: Input text or list of texts.
            pooling_method: Pooling method ('mean', 'last_token', 'eos_token').
                          If None, defaults to 'last_token' when prompt_template is specified,
                          otherwise defaults to 'mean'.
            layer_index: Layer index to extract embeddings from.
                        If None, defaults to -2 for 'pcoteol'/'ke' templates, -1 otherwise.
                        Note: vLLM backend typically only supports the last layer (-1).
            prompt_template: Optional prompt template ('prompteol', 'pcoteol', 'ke').
                           When specified, wraps the input text with the template.
            batch_size: Batch size for processing. If None, processes all inputs at once.
            **kwargs: Backend specific arguments.

        Returns:
            Embeddings as numpy array or torch tensor.
        """
        # Smart default: use last_token pooling when template is provided
        if pooling_method is None:
            if prompt_template is not None:
                pooling_method = "last_token"
            else:
                pooling_method = "mean"
        
        if isinstance(text, str):
            text = [text]

        if batch_size is None:
            return self.backend_instance.encode(
                text, pooling_method=pooling_method, layer_index=layer_index, 
                prompt_template=prompt_template, **kwargs
            )

        results = []
        total = len(text)
        
        for i in tqdm(range(0, total, batch_size), desc="Encoding", disable=total <= batch_size):
            batch_text = text[i : i + batch_size]
            batch_emb = self.backend_instance.encode(
                batch_text, pooling_method=pooling_method, layer_index=layer_index,
                prompt_template=prompt_template, **kwargs
            )
            # Ensure it's a tensor for concatenation logic
            if not isinstance(batch_emb, torch.Tensor):
                 batch_emb = torch.tensor(batch_emb)
            
            # Offload to CPU immediately to prevent VRAM accumulation
            batch_emb = batch_emb.detach().cpu()
            
            results.append(batch_emb)
            
        return torch.cat(results, dim=0)
