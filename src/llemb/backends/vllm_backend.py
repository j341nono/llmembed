import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from vllm import LLM, PoolingParams

from ..interfaces import Backend

logger = logging.getLogger(__name__)

class VLLMBackend(Backend):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        **kwargs: Any
    ):
        """
        Initialize VLLMBackend.
        
        Args:
            model_name: HuggingFace model identifier.
            device: Device (vLLM usually requires 'cuda').
            quantization: Quantization config (e.g., 'fp8', 'awq', 'gptq', 'bitsandbytes').
            gpu_memory_utilization: vLLM argument.
            max_model_len: Context length.
            tensor_parallel_size: Number of GPUs.
            **kwargs: Additional arguments passed to LLM init.
        """
        self.model_name = model_name
        self.device = device
        
        # vLLM arguments preparation
        vllm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "quantization": quantization,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            # runner="pooling" is implicitly handled if task is specified in recent vLLM, 
            # but explicit setting helps.
            "enforce_eager": kwargs.pop("enforce_eager", False), 
        }
        
        if max_model_len:
            vllm_kwargs["max_model_len"] = max_model_len

        # Update with any remaining kwargs
        vllm_kwargs.update(kwargs)

        logger.info(f"Initializing vLLM with args: {vllm_kwargs}")
        
        # Initialize vLLM Engine
        # Note: vLLM usually expects to be initialized once per process.
        self.model = LLM(**vllm_kwargs)
        
        # Get tokenizer for EOS token handling
        self.tokenizer = self.model.get_tokenizer()

    def encode(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        layer_index: Optional[int] = None,
        **kwargs: Any
    ) -> Union["np.ndarray[Any, Any]", torch.Tensor]:
        """
        Encode text using vLLM using 'token_embed' task to fetch all token embeddings,
        then apply pooling logic client-side.
        """
        if self.model is None:
            raise RuntimeError("vLLM Model not initialized")

        # Warning about layer_index limitation in vLLM
        if layer_index is not None and layer_index != -1:
            logger.warning(
                f"layer_index={layer_index} was requested, but vLLM standardly returns "
                "embeddings from the last layer (or the target embedding layer). "
                "This parameter might be ignored unless the model supports layer selection natively."
            )

        if isinstance(text, str):
            text = [text]

        if not text:
            return torch.empty(0)

        # --- Prompt Engineering (Same as TransformersBackend) ---
        original_texts = text # Keep original for reference if needed
        prompts = []
        
        if pooling == "prompt_eol":
            prompts = [f'This Sentence : "{t}" means in one word:"' for t in text]
        elif pooling == "pcoteol":
            prompts = [
                f'After thinking step by step, this sentence : "{t}" means in one word:"'
                for t in text
            ]
        elif pooling == "ke":
            prompts = [
                f'The essence of a sentence is often captured by its main subjects and actions, '
                f'while descriptive terms provide additional but less central details. '
                f'With this in mind , this sentence : "{t}" means in one word:"' 
                for t in text
            ]
        else:
            prompts = text

        # --- vLLM Execution ---
        # Request 'token_embed' task to get [SeqLen, Hidden] for each prompt.
        # This allows us to calculate arbitrary indices manually.
        pooling_params = PoolingParams(task="token_embed")
        
        # embed() returns List[EmbeddingRequestOutput]
        # use_tqdm=False to reduce noise
        outputs = self.model.embed(prompts, pooling_params=pooling_params, use_tqdm=False)
        
        # --- Client-side Pooling Logic ---
        embeddings_list = []
        
        for i, output in enumerate(outputs):
            # output.outputs.embedding is List[float] (flat) or List[List[float]]?
            # In 'token_embed' task, it should be a sequence of embeddings.
            # Convert to tensor: shape [SeqLen, HiddenDim]
            token_embeddings = torch.tensor(output.outputs.embedding, device=self.device)
            
            # vLLM returns embeddings for non-padding tokens.
            # We don't need attention_mask here because vLLM output corresponds to actual tokens.
            seq_len = token_embeddings.size(0)
            
            # Apply Pooling
            if pooling == "mean":
                # Average over Sequence dimension
                emb = torch.mean(token_embeddings, dim=0)
                
            elif pooling == "last_token":
                # Last token in the sequence
                emb = token_embeddings[-1]
                
            elif pooling == "eos_token":
                # Re-tokenize to find EOS position matches
                # Note: This is computationally expensive but necessary since vLLM outputs
                # don't carry token IDs in the embedding output object by default in all versions.
                # However, output.prompt_token_ids is available!
                token_ids = output.prompt_token_ids
                eos_id = self.tokenizer.eos_token_id
                
                # Find the last occurrence of EOS
                # indices where token_ids == eos_id
                indices = [idx for idx, tid in enumerate(token_ids) if tid == eos_id]
                
                if indices:
                    target_idx = indices[-1]
                    emb = token_embeddings[target_idx]
                else:
                    logger.warning(
                        f"EOS token not found for sequence at index {i}. "
                        "Falling back to last token."
                    )
                    emb = token_embeddings[-1]

            elif pooling in ["prompt_eol", "pcoteol", "ke"]:
                # These methods target the last token of the constructed prompt (which ends in quote)
                emb = token_embeddings[-1]
            
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            
            embeddings_list.append(emb)

        # Stack and move to CPU
        if not embeddings_list:
            return torch.empty(0)
            
        final_embeddings = torch.stack(embeddings_list)
        return final_embeddings.cpu().detach()