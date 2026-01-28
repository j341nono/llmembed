import logging
from typing import Any, List, Optional, Union

import numpy as np
import torch

from ..interfaces import Backend

try:
    from vllm import LLM, PoolingParams
except ImportError:
    LLM = None
    PoolingParams = None

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
        **kwargs: Any,
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
            **kwargs: Additional arguments passed to LLM init (e.g. enforce_eager).
        """
        if LLM is None or PoolingParams is None:
            raise ImportError(
                "VLLMBackend requires 'vllm'. "
                "Please install it with `pip install vllm` or `pip install llemb[vllm]`."
            )

        self.model_name = model_name
        self.device = device

        enforce_eager = kwargs.pop("enforce_eager", True)

        vllm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "quantization": quantization,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": enforce_eager,
            "runner": "pooling",
        }

        if max_model_len:
            vllm_kwargs["max_model_len"] = max_model_len

        vllm_kwargs.update(kwargs)

        logger.info(f"Initializing vLLM with args: {vllm_kwargs}")

        self.model = LLM(**vllm_kwargs)
        self.tokenizer = self.model.get_tokenizer()

    def encode(
        self,
        text: Union[str, List[str]],
        pooling_method: Optional[str] = None,
        layer_index: Optional[int] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["np.ndarray[Any, Any]", torch.Tensor]:
        """
        Encode text using vLLM.
        We request 'token_embed' task to fetch all token embeddings,
        then apply pooling logic client-side.
        """
        if self.model is None:
            raise RuntimeError("vLLM Model not initialized")

        # Smart default: use last_token pooling when template is provided
        if pooling_method is None:
            if prompt_template is not None:
                pooling_method = "last_token"
            else:
                pooling_method = "mean"

        # vLLM backend warning for layer_index
        if layer_index is not None and layer_index != -1:
            logger.warning(
                f"layer_index={layer_index} was requested, but vLLM backend currently "
                "only supports the last layer output. This parameter is ignored."
            )

        if isinstance(text, str):
            text = [text]

        if not text:
            return torch.empty(0)

        # Apply prompt template if specified
        prompts = []
        if prompt_template == "prompteol":
            prompts = [f'This Sentence : "{t}" means in one word:"' for t in text]
        elif prompt_template == "pcoteol":
            prompts = [
                f'After thinking step by step, this sentence : "{t}" means in one word:"'
                for t in text
            ]
        elif prompt_template == "ke":
            prompts = [
                f"The essence of a sentence is often captured by its main subjects and actions, "
                f"while descriptive terms provide additional but less central details. "
                f'With this in mind , this sentence : "{t}" means in one word:"'
                for t in text
            ]
        elif prompt_template is not None:
            raise ValueError(f"Unknown prompt_template: {prompt_template}")
        else:
            prompts = text

        try:
            pooling_params = PoolingParams(task="token_embed")
            outputs = self.model.encode(
                prompts, pooling_params=pooling_params, use_tqdm=False, pooling_task="token_embed"
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(
                f"Failed to use LLM.encode with token_embed: {e}. Falling back to LLM.embed."
            )
            pooling_params = PoolingParams()
            outputs = self.model.embed(prompts, pooling_params=pooling_params, use_tqdm=False)

        embeddings_list = []

        for i, output in enumerate(outputs):
            if hasattr(output, "outputs"):
                out_data = output.outputs
            else:
                out_data = output

            if hasattr(out_data, "embedding"):
                raw_emb = out_data.embedding
            elif hasattr(out_data, "data"):
                raw_emb = out_data.data
            else:
                raw_emb = getattr(out_data, "embedding", [])

            if isinstance(raw_emb, torch.Tensor):
                val = raw_emb.to(self.device)
            else:
                val = torch.tensor(raw_emb, device=self.device)

            if val.ndim == 1:
                token_embeddings = val.unsqueeze(0)
            else:
                token_embeddings = val

            # Apply pooling method
            if pooling_method == "mean":
                if token_embeddings.size(0) > 1:
                    emb = torch.mean(token_embeddings, dim=0)
                else:
                    emb = token_embeddings[0]

            elif pooling_method == "last_token":
                emb = token_embeddings[-1]

            elif pooling_method == "eos_token":
                if hasattr(output, "prompt_token_ids"):
                    token_ids = output.prompt_token_ids
                    eos_id = self.tokenizer.eos_token_id

                    indices = [idx for idx, tid in enumerate(token_ids) if tid == eos_id]

                    if indices and indices[-1] < token_embeddings.size(0):
                        emb = token_embeddings[indices[-1]]
                    else:
                        logger.debug(f"EOS token not found for sequence {i}, using last token.")
                        emb = token_embeddings[-1]
                else:
                    emb = token_embeddings[-1]

            else:
                raise ValueError(f"Unknown pooling_method: {pooling_method}")

            embeddings_list.append(emb)

        if not embeddings_list:
            return torch.empty(0)

        return torch.stack(embeddings_list).cpu().detach()