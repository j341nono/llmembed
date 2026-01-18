import sys
from llmembed import Encoder

def test_transformers():
    print("Testing Transformers Backend...")
    enc = Encoder(
        model_name="sshleifer/tiny-gpt2",
        backend="transformers",
        device="cpu"
    )
    try:
        emb = enc.encode("Hello world")
        print(f"Transformers embedding shape: {emb.shape}")
        assert emb.shape[0] == 1
    except Exception as e:
        print(f"Transformers Test Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_transformers()
