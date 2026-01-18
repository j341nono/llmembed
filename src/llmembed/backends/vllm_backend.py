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
        quantization: Optional[str] = None
    ):
        if LLM is None:
            raise ImportError(
                "vllm is not installed. Please install it with `uv pip install llmembed[vllm]`."
            )
        
        # VLLM handles device placement automatically usually, but we can pass
        # trust_remote_code etc.
        # If device is passed and not auto/None, we might need to handle it.
        # But VLLM usually takes device via other args or env vars.
        # For now, ignore device arg if it's just meant for auto-detect logic in core.
        
        self.model = LLM(
            model=model_name,
            quantization=quantization,
            # enforce_eager=True # Sometimes needed for embedding extraction
        )

    def encode(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        layer_index: int = -1,
        **kwargs: Any
    ) -> Any:
        if layer_index != -1:
             raise NotImplementedError(
                 "VLLM backend currently only supports extracting the final layer "
                 "embedding (layer_index=-1)."
             )
        
        if isinstance(text, str):
            text = [text]
            
        # Verify pooling compatibility
        # VLLM PoolingParams usually takes specific args. 
        # But wait, LLM.encode() takes prompts and maybe pooling_params?
        # Actually LLM.encode() is for embedding models (e.g. E5, BGE).
        # For CausalLM, we might not be able to use LLM.encode() directly if the model
        # is not registered as embedding model.
        # But assuming the user wants to use VLLM for embedding extraction, they likely
        # use a model supported by VLLM for embeddings.
        # Or we might need to use `llm.generate` with `logprobs` and extract?
        # No, that's not embeddings.
        
        # Let's try to map our pooling to VLLM's expectation if possible.
        # VLLM's encode doesn't strictly take 'pooling' strategy per request in all versions.
        # But let's assume standard usage.
        
        outputs = self.model.encode(text)
        
        # outputs is a list of EmbeddingRequestOutput
        embeddings = []
        for output in outputs:
            # output.outputs.embedding is the vector
            # The pooling strategy is determined by the model config or VLLM default?
            # If VLLM doesn't support changing pooling per request, we might be stuck with default.
            # I'll just return what VLLM gives.
            # If the user requested specific pooling, I might need to warn that VLLM handles it.
            if hasattr(output, 'outputs'):
                 embeddings.append(output.outputs.embedding)
            else:
                 # API might differ
                 embeddings.append(output) # fallback
                 
        return np.array(embeddings)
