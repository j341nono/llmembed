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

def test_vllm():
    print("\nTesting VLLM Backend...")
    try:
        import vllm
    except ImportError:
        print("vLLM not installed, skipping test.")
        return

    # Explicit memory configuration for testing environment
    try:
        enc = Encoder(
            model_name="sshleifer/tiny-gpt2",
            backend="vllm",
            gpu_memory_utilization=0.4, # Explicitly set for testing environment
            max_model_len=1024,
        )
        emb = enc.encode("Hello world")
        print(f"VLLM embedding shape: {emb.shape}")
        assert emb.shape[0] == 1
    except Exception as e:
        print(f"VLLM Test Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_transformers()
    test_vllm()
