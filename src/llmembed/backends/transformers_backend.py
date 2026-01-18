from typing import Any, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..interfaces import Backend


class TransformersBackend(Backend):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        quantization: Optional[str] = None
    ):
        """
        Initialize TransformersBackend.
        
        Args:
            model_name: HuggingFace model identifier.
            device: Device to load model on ('cpu', 'cuda', 'mps').
            quantization: Quantization config ('bitsandbytes', '4bit', '8bit', None).
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        quantization_config = None
        load_kws = {}
        
        if self.quantization:
            if self.quantization in ["bitsandbytes", "4bit"]:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True) # type: ignore
            elif self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True) # type: ignore
            
            if quantization_config:
                load_kws["quantization_config"] = quantization_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) # type: ignore
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
        layer_index: int = -1,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not initialized")

        if isinstance(text, str):
            text = [text]
            
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

        elif pooling == "prompt_eol":
            # Find last newline token.
            # Simple heuristic: decode and find '\n' char position? 
            # Better: Search for tokens that represent '\n'. 
            # Or iterate tokens?
            # Since tokenization is complex, maybe just decoding is slow but accurate?
            # But we need token index.
            # Strategy: Get offset mapping? 
            # Simplified: Look for a specific token ID? '\n' might be multiple IDs.
            # Let's assume we want the last token that is NOT padding...
            # The requirement: "providing a prompt and getting the embedding
            # of the token immediately following the prompt."
            # If I stick to my interpretation: Find the last '\n' in the sequence.
            
            # Alternative interpretation based on "immediately following the prompt":
            # Maybe the user provides `prompt` and `text` (completion)?
            # But the signature is `encode(text)`.
            # I will implement finding the last '\n' character's token.
            
            # This is tricky with BPE. '\n' might be part of a word.
            # I'll implement a heuristic: Find the last token whose string
            # representation ends with '\n'.
            
            embeddings_list = []
            for i in range(input_ids.size(0)):
                tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[i])
                found = False
                # Search backwards from last non-pad token
                last_idx = (inputs.attention_mask[i].sum() - 1).item()
                for idx in range(last_idx, -1, -1):
                    token_str = tokens[idx]
                    # Clean up token string (transformers often adds Ġ or similar)
                    if hasattr(self.tokenizer, "convert_tokens_to_string"):
                         decoded = self.tokenizer.convert_tokens_to_string([token_str])
                    else:
                         decoded = token_str # fallback
                    
                    if "\n" in decoded:
                        embeddings_list.append(hidden_states[i, idx])
                        found = True
                        break
                if not found:
                    embeddings_list.append(hidden_states[i, last_idx])
            embeddings = torch.stack(embeddings_list)
            
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
            
        return embeddings.cpu().detach()
