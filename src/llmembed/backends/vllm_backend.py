from typing import Any, List, Optional, Union

import numpy as np

try:
    from vllm import LLM, PoolingParams
except ImportError:
    LLM = None
    PoolingParams = None

from ..interfaces import Backend


class VLLMBackend(Backend):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None, # Was str="auto"
        quantization: Optional[str] = None,
        **kwargs: Any
    ):
        if LLM is None:
            raise ImportError(
                "vllm is not installed. Please install it with `uv pip install llmembed[vllm]`."
            )
        
        # User is responsible for memory config via kwargs (gpu_memory_utilization, etc.)
        self.model = LLM(
            model=model_name,
            quantization=quantization,
            trust_remote_code=True,
            **kwargs
        )

    def encode(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        layer_index: Optional[int] = None,
        **kwargs: Any
    ) -> Any:
        # Default layer_index logic
        if layer_index is None:
             layer_index = -1

        if layer_index != -1:
             raise NotImplementedError(
                 "VLLM backend currently only supports extracting the final layer "
                 "embedding (layer_index=-1)."
             )
        
        if isinstance(text, str):
            text = [text]
            
        # VLLM encode logic
        outputs = self.model.encode(text)
        
        # outputs is a list of EmbeddingRequestOutput
        embeddings = []
        for output in outputs:
            if hasattr(output, 'outputs'):
                 embeddings.append(output.outputs.embedding)
            else:
                 # API might differ
                 embeddings.append(output) # fallback
                 
        return np.array(embeddings)
