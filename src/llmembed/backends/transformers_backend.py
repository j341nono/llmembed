from typing import Any, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..interfaces import Backend


class TransformersBackend(Backend):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        quantization: Optional[str] = None
    ):
        """
        Initialize TransformersBackend.
        
        Args:
            model_name: HuggingFace model identifier.
            device: Device to load model on ('cpu', 'cuda', 'mps'). If None, auto-detects.
            quantization: Quantization config ('4bit', '8bit', or None).
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
        self._load_model()

    def _load_model(self) -> None:
        quantization_config = None
        load_kws = {}
        
        if self.quantization:
            if self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            if quantization_config:
                load_kws["quantization_config"] = quantization_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        assert self.tokenizer is not None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kws
        )
        
        if not self.quantization:
             assert self.model is not None
             self.model.to(self.device)

    def encode(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        layer_index: Optional[int] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not initialized")

        # Set default layer_index based on pooling strategy
        if layer_index is None:
            if pooling in ["pcoteol", "ke"]:
                layer_index = -2
            else:
                layer_index = -1
        
        if isinstance(text, str):
            text = [text]
            
        if pooling == "prompt_eol":
            text = [f'This Sentence : "{t}" means in one word:"' for t in text]
        elif pooling == "pcoteol":
            text = [
                f'After thinking step by step, this sentence : "{t}" means in one word:"'
                for t in text
            ]
        elif pooling == "ke":
            text = [
                f'The essence of a sentence is often captured by its main subjects and actions, '
                f'while descriptive terms provide additional but less central details. '
                f'With this in mind , this sentence : "{t}" means in one word:"' 
                for t in text
            ]
            
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Get hidden states
        # outputs.hidden_states is a tuple of (layer_0, ..., layer_N)
        # layer_index -1 means last layer.
        # Note: hidden_states usually includes embeddings at index 0? 
        # Transformers output_hidden_states=True returns (embeddings, layer_1, ... layer_N)
        # So len is num_layers + 1.
        # layer_index -1 is the last one.
        
        hidden_states = outputs.hidden_states[layer_index]
        
        input_ids = inputs.input_ids
        
        if pooling == "mean":
            # Mask padding tokens
            mask = inputs.attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
        elif pooling == "last_token":
            # Use attention_mask to find the last non-padding token
            # inputs.attention_mask: [batch, seq_len] with 1 for token, 0 for pad
            # We want the index of the last '1'.
            # Note: left-padding vs right-padding matters. 
            # AutoTokenizer usually right-pads by default for some, left for others.
            # We can find the last index where mask is 1.
            seq_lengths = inputs.attention_mask.sum(dim=1) - 1
            # Gather
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            embeddings = hidden_states[batch_indices, seq_lengths]
            
        elif pooling == "eos_token":
             # Find index of EOS token.
             # If multiple EOS, taking the last one? Or the first one after text?
             # Assuming inputs might contain EOS.
             # If no EOS found, fallback to last_token?
             # For now, simplistic implementation: find last occurrence of eos_token_id
             eos_id = self.tokenizer.eos_token_id
             input_ids = inputs.input_ids
             
             # Create a mask where input_ids == eos_id
             matches = (input_ids == eos_id)
             # If any match
             embeddings_list = []
             for i in range(input_ids.size(0)):
                 row_matches = matches[i].nonzero()
                 if row_matches.size(0) > 0:
                     last_eos_idx = row_matches[-1].item()
                     embeddings_list.append(hidden_states[i, last_eos_idx])
                 else:
                     # Fallback to last token if no EOS found
                     last_idx = inputs.attention_mask[i].sum() - 1
                     embeddings_list.append(hidden_states[i, last_idx])
             embeddings = torch.stack(embeddings_list)

        elif pooling in ["prompt_eol", "pcoteol", "ke"]:
            # Extract the very last token (corresponding to the final ")
            # Use attention_mask to find the last non-padding token
            seq_lengths = inputs.attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            embeddings = hidden_states[batch_indices, seq_lengths]
            
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
            
        return embeddings.cpu().detach()
