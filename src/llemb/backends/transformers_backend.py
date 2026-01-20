import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..interfaces import Backend

logger = logging.getLogger(__name__)


class TransformersBackend(Backend):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize TransformersBackend.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to load model on ('cpu', 'cuda', 'mps'). If None, auto-detects.
            quantization: Quantization config ('4bit', '8bit', or None).
            **kwargs: Additional arguments passed to the backend (e.g., model_kwargs).
        """
        self.model_name = model_name
        self.quantization = quantization

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None

        if "model_kwargs" in kwargs:
            kwargs.update(kwargs.pop("model_kwargs"))

        self._load_model(kwargs)

    def _load_model(self, load_kws: "Dict[str, Any]") -> None:
        quantization_config = None
        load_kws = load_kws.copy()

        if self.quantization:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "Quantization requires 'bitsandbytes'. "
                    "Please install it with `pip install llemb[quantization]`."
                )

            if self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            if quantization_config:
                load_kws["quantization_config"] = quantization_config

            if "device_map" not in load_kws:
                load_kws["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        assert self.tokenizer is not None
        self.tokenizer.padding_side = 'left' # Force left padding for correct generation/pooling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kws)

        if not self.quantization and "device_map" not in load_kws:
            assert self.model is not None
            self.model.to(self.device)

    def encode(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        layer_index: Optional[int] = None,
        **kwargs: Any,
    ) -> Union["np.ndarray[Any, Any]", torch.Tensor]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not initialized")

        if layer_index is None:
            if pooling in ["pcoteol", "ke"]:
                layer_index = -2
            else:
                layer_index = -1

        if isinstance(text, str):
            text = [text]

        if not text:
            return torch.empty(0)

        if pooling == "prompt_eol":
            text = [f'This Sentence : "{t}" means in one word:"' for t in text]
        elif pooling == "pcoteol":
            text = [
                f'After thinking step by step, this sentence : "{t}" means in one word:"'
                for t in text
            ]
        elif pooling == "ke":
            text = [
                f"The essence of a sentence is often captured by its main subjects and actions, "
                f"while descriptive terms provide additional but less central details. "
                f'With this in mind , this sentence : "{t}" means in one word:"'
                for t in text
            ]

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        num_layers = len(outputs.hidden_states)
        if not (-num_layers <= layer_index < num_layers):
            raise ValueError(
                f"layer_index {layer_index} is out of bounds. "
                f"Model has {num_layers} layers (valid indices: {-num_layers} to {num_layers - 1})."
            )

        hidden_states = outputs.hidden_states[layer_index]
        if pooling == "mean":
            mask = inputs.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        elif pooling == "last_token":
            # With left padding, the last token is always at index -1
            embeddings = hidden_states[:, -1, :]

        elif pooling == "index":
            target_idx = kwargs.get("token_index", -1)
            batch_size = hidden_states.size(0)
            seq_len = hidden_states.size(1)

            if not (-seq_len <= target_idx < seq_len):
                logger.warning(
                    f"Requested token_index={target_idx} is out of bounds "
                    f"for sequence length {seq_len}. Falling back to last token."
                )
                # Fallback to last token (simplified for left padding)
                embeddings = hidden_states[:, -1, :]
            else:
                embeddings = hidden_states[:, target_idx, :]

        elif pooling == "eos_token":
            # Use attention_mask to find the last non-padding token
            # With left padding, the last token is also the effective EOS token
            # unless the generation stopped early (not applicable for embedding extraction)
            # We can robustly use index -1 if we trust left padding + truncation
            embeddings = hidden_states[:, -1, :]

        elif pooling in ["prompt_eol", "pcoteol", "ke"]:
            # Extract the very last token (corresponding to the final ")
            # With left padding, this is simply index -1
            embeddings = hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        return embeddings.cpu().detach()
