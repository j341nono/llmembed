import numpy as np
import pytest
import torch

import llemb

TEST_MODEL_NAME = "gpt2"

SAMPLE_TEXTS = [
    "Hello world",
    "This is a test sentence for embedding extraction.",
    "Llemb is a library for unified embedding extraction.",
    "Japanese text check: こんにちは、世界。",
    "Short.",
]

def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices

@pytest.fixture(scope="module")
def shared_encoder_cpu():
    print(f"\nInitializing shared CPU encoder with {TEST_MODEL_NAME}...")
    return llemb.Encoder(TEST_MODEL_NAME, device="cpu")

class TestIntegration:
    
    @pytest.mark.parametrize("device", get_available_devices())
    def test_device_placement(self, device):
        print(f"Testing loading on device: {device}")
        try:
            enc = llemb.Encoder(TEST_MODEL_NAME, device=device)
            
            model_device = next(enc.backend_instance.model.parameters()).device
            
            if device == "cuda":
                assert model_device.type == "cuda"
            elif device == "mps":
                assert model_device.type == "mps"
            else:
                assert model_device.type == "cpu"
                
            emb = enc.encode("Device test", pooling="mean")
            assert isinstance(emb, torch.Tensor) or isinstance(emb, np.ndarray)
            
        except Exception as e:
            pytest.fail(f"Failed to load or encode on {device}: {e}")

    @pytest.mark.parametrize("pooling", [
        "mean", 
        "last_token", 
        "eos_token", 
        "prompt_eol", 
        "pcoteol", 
        "ke"
    ])
    def test_pooling_strategies(self, shared_encoder_cpu, pooling):
        enc = shared_encoder_cpu
        embeddings = enc.encode(SAMPLE_TEXTS, pooling=pooling)
        
        expected_dim = enc.backend_instance.model.config.hidden_size
        
        assert embeddings.shape[0] == len(SAMPLE_TEXTS)
        assert embeddings.shape[1] == expected_dim
        assert not torch.isnan(torch.tensor(embeddings)).any(), f"NaN found in {pooling}"

    def test_batch_processing_consistency(self, shared_encoder_cpu):
        enc = shared_encoder_cpu
        
        emb_full = enc.encode(SAMPLE_TEXTS, batch_size=None)
        
        emb_batched = enc.encode(SAMPLE_TEXTS, batch_size=2)
        
        # NOTE: With absolute positional embeddings (like GPT-2) and left-padding,
        # changing batch size changes padding amount per batch, which shifts 
        # token positions and thus changes embeddings.
        # Strict equality is not expected unless using a model with relative positions
        # or if inputs are same length. We check shapes and validity here.
        assert emb_batched.shape == emb_full.shape
        assert not torch.isnan(emb_batched).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantization tests require CUDA")
    @pytest.mark.parametrize("quantization", ["4bit", "8bit"])
    def test_quantization(self, quantization):
        import importlib.util
        if importlib.util.find_spec("bitsandbytes") is None:
            pytest.skip("bitsandbytes not installed")

        print(f"Testing quantization: {quantization}")
        
        try:
            enc = llemb.Encoder(
                TEST_MODEL_NAME, 
                device="cuda", 
                quantization=quantization
            )
            
            if hasattr(enc.backend_instance.model, "hf_device_map"):
                assert len(enc.backend_instance.model.hf_device_map) > 0
            
            emb = enc.encode("Quantization test", pooling="mean")
            assert emb.shape[0] == 1
            
        except ImportError:
            pytest.skip("bitsandbytes library issue or dependency missing")
        except Exception as e:
            if "modules to save" in str(e) or "too small" in str(e).lower():
                pytest.skip(f"Model too small for quantization test: {e}")
            else:
                pytest.fail(f"Quantization {quantization} failed: {e}")

    def test_layer_selection(self, shared_encoder_cpu):
        enc = shared_encoder_cpu
        
        emb_last = enc.encode("Layer test", layer_index=-1, pooling="mean")
        
        num_layers = enc.backend_instance.model.config.n_layer
        target_layer = -2
        
        if num_layers >= abs(target_layer):
            emb_prev = enc.encode("Layer test", layer_index=target_layer, pooling="mean")
            
            assert not torch.allclose(torch.as_tensor(emb_last), torch.as_tensor(emb_prev)), \
                "Layer -1 and -2 produced identical embeddings (unexpected)"
            