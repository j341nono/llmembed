import pytest

from llemb.backends.transformers_backend import TransformersBackend
from llemb.backends.vllm_backend import VLLMBackend
from llemb.core import Encoder


def test_encoder_init_transformers():
    enc = Encoder(model_name="sshleifer/tiny-gpt2", backend="transformers")
    assert isinstance(enc.backend_instance, TransformersBackend)

def test_encoder_init_vllm():
    enc = Encoder(model_name="sshleifer/tiny-gpt2", backend="vllm")
    assert isinstance(enc.backend_instance, VLLMBackend)

def test_encoder_invalid_backend():
    with pytest.raises(ValueError):
        Encoder(model_name="sshleifer/tiny-gpt2", backend="invalid")

def test_encoder_encode_transformers():
    enc = Encoder(model_name="sshleifer/tiny-gpt2", backend="transformers")
    res = enc.encode("hello")
    assert res is not None

def test_encoder_encode_vllm():
    enc = Encoder(model_name="sshleifer/tiny-gpt2", backend="vllm")
    res = enc.encode("hello")
    assert res is not None
