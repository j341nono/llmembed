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
        pooling_method: Optional[str] = None,
        layer_index: Optional[int] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["np.ndarray[Any, Any]", torch.Tensor]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not initialized")

        # Smart default: use last_token pooling when template is provided
        if pooling_method is None:
            if prompt_template is not None:
                pooling_method = "last_token"
            else:
                pooling_method = "mean"

        # Determine layer index with priority: explicit > template-specific > global default
        if layer_index is None:
            if prompt_template in ["pcoteol", "ke"]:
                layer_index = -2
            else:
                layer_index = -1

        if isinstance(text, str):
            text = [text]

        if not text:
            return torch.empty(0)

        # Apply prompt template if specified
        if prompt_template == "prompteol":
            text = [f'This Sentence : "{t}" means in one word:"' for t in text]
        elif prompt_template == "pcoteol":
            text = [
                f'After thinking step by step, this sentence : "{t}" means in one word:"'
                for t in text
            ]
        elif prompt_template == "ke":
            text = [
                f"The essence of a sentence is often captured by its main subjects and actions, "
                f"while descriptive terms provide additional but less central details. "
                f'With this in mind , this sentence : "{t}" means in one word:"'
                for t in text
            ]
        elif prompt_template is not None:
            raise ValueError(f"Unknown prompt_template: {prompt_template}")

        # For eos_token pooling, ensure EOS token is present in the input
        if pooling_method == "eos_token" and self.tokenizer.eos_token:
            # Check and append EOS token if not present
            text = [
                t if t.endswith(self.tokenizer.eos_token) else t + self.tokenizer.eos_token
                for t in text
            ]

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        num_layers = len(outputs.hidden_states)
        if not (-num_layers <= layer_index < num_layers):
            raise ValueError(
                f"layer_index {layer_index} is out of bounds. "
                f"Model has {num_layers} layers (valid indices: {-num_layers} to {num_layers - 1})."
            )

        hidden_states = outputs.hidden_states[layer_index]
        
        # Apply pooling method
        if pooling_method == "mean":
            mask = inputs.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        elif pooling_method == "last_token":
            # With left padding, the last token is always at index -1
            embeddings = hidden_states[:, -1, :]

        elif pooling_method == "eos_token":
            # Use attention_mask to find the last non-padding token (which should be EOS)
            # With left padding, we need to find the actual position of the last real token
            batch_size = hidden_states.size(0)
            seq_lengths = inputs.attention_mask.sum(dim=1)  # Length of each sequence
            
            # Extract embedding at the last non-padded position for each item in batch
            # Since we use left padding, the last real token is at position (seq_length - 1)
            # But with left padding at position [-1], this is always the last token
            embeddings = hidden_states[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling_method: {pooling_method}")

        return embeddings.cpu().detach()
