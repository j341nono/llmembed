#!/usr/bin/env python3
"""
Comprehensive GPU Verification Script for llemb v0.2.2
=======================================================

This script verifies that the llemb library works correctly in a GPU environment
with both Transformers and vLLM backends. It tests:
- Quantization with bitsandbytes
- New Smart Default API (v0.2.2)
- Mixed precision handling
- vLLM backend integration

Requirements:
- CUDA-capable GPU
- bitsandbytes library (core dependency in v0.2.2)
- torch with CUDA support
- vllm (optional, will skip if not installed)
"""

import gc
import sys
from typing import Optional


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"✗ {message}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ {message}")


def cleanup_gpu_memory() -> None:
    """
    Aggressively clean up GPU memory.
    
    This is critical before loading vLLM to prevent OOM errors,
    as vLLM reserves a large amount of GPU memory on initialization.
    """
    try:
        import torch
        
        # Delete any lingering references
        if 'enc' in globals():
            del globals()['enc']
        if 'model' in globals():
            del globals()['model']
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print_info("GPU memory cleanup completed")
        
    except Exception as e:
        print_warning(f"GPU cleanup had issues (non-critical): {e}")


def check_environment() -> bool:
    """
    Check if the environment is ready for GPU testing.
    
    Returns:
        True if environment is ready, False otherwise.
    """
    print_header("Step 1: Environment & CUDA Check")
    
    # Check torch
    try:
        import torch
        print_success("PyTorch is installed")
    except ImportError as e:
        print_error(f"Failed to import torch: {e}")
        return False
    
    # Check CUDA
    if not torch.cuda.is_available():
        print_error("CUDA is NOT available")
        print_info("This script requires a CUDA-capable GPU")
        print_info("Please ensure:")
        print_info("  1. You have a CUDA-capable GPU")
        print_info("  2. NVIDIA drivers are installed")
        print_info("  3. PyTorch is installed with CUDA support")
        return False
    
    cuda_version = torch.version.cuda
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    
    print_success(f"CUDA is available (version: {cuda_version})")
    print_info(f"Detected {device_count} CUDA device(s)")
    print_info(f"Primary device: {device_name}")
    
    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        print_success("bitsandbytes is installed")
        
        try:
            bnb_version = bnb.__version__
            print_info(f"bitsandbytes version: {bnb_version}")
        except AttributeError:
            print_info("bitsandbytes version: unknown")
            
    except ImportError:
        print_error("bitsandbytes is NOT installed")
        print_info("As of v0.2.2, bitsandbytes is a core dependency")
        print_info("Install with: pip install bitsandbytes>=0.48.2")
        return False
    
    # Check llemb
    try:
        import llemb
        print_success("llemb is installed")
    except ImportError:
        print_error("llemb is NOT installed")
        print_info("Install with: pip install -e .")
        return False
    
    return True


def test_transformers_backend(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    quantization: str = "4bit"
) -> bool:
    """
    Test Transformers backend with quantization and Smart Defaults.
    
    Args:
        model_name: HuggingFace model identifier
        quantization: Quantization mode ('4bit' or '8bit')
    
    Returns:
        True if all tests pass, False otherwise.
    """
    print_header(f"Step 2: Transformers Backend Test ({quantization.upper()} Quantization)")
    
    try:
        import torch
        import llemb
        
        print_info(f"Loading model: {model_name}")
        print_info(f"Quantization: {quantization}")
        print_info("This may take a moment...")
        
        # Initialize encoder with quantization
        enc = llemb.Encoder(
            model_name=model_name,
            backend="transformers",
            device="cuda",
            quantization=quantization
        )
        
        print_success(f"Model loaded with {quantization} quantization")
        
        # Get model device info
        model_device = next(enc.backend_instance.model.parameters()).device
        print_info(f"Model is on device: {model_device}")
        
        # Test Smart Default API
        print()
        print_info("Testing Smart Default API (v0.2.2)")
        
        test_texts = [
            "GPU Test with Transformers",
            "Testing quantized inference",
            "Batch processing test"
        ]
        
        # Test 1: Smart Default with pcoteol
        print_info("Test 1: prompt_template='pcoteol' (auto last_token)")
        emb1 = enc.encode(test_texts[0], prompt_template="pcoteol")
        print_success(f"Embedding: shape={emb1.shape}, dtype={emb1.dtype}")
        
        # Test 2: Smart Default with ke
        print_info("Test 2: prompt_template='ke' (auto last_token)")
        emb2 = enc.encode(test_texts[1], prompt_template="ke")
        print_success(f"Embedding: shape={emb2.shape}, dtype={emb2.dtype}")
        
        # Test 3: Batch processing
        print_info("Test 3: Batch processing")
        emb_batch = enc.encode(test_texts, prompt_template="prompteol")
        print_success(f"Batch: shape={emb_batch.shape}, dtype={emb_batch.dtype}")
        
        # Validate embeddings
        if torch.isnan(emb_batch).any():
            print_error("Embeddings contain NaN values")
            return False
        
        if torch.isinf(emb_batch).any():
            print_error("Embeddings contain Inf values")
            return False
        
        print_success("All embeddings are valid")
        
        # Test 4: Explicit override
        print_info("Test 4: Explicit pooling_method override")
        emb_mean = enc.encode(
            test_texts[0],
            pooling_method="mean",
            prompt_template="pcoteol"
        )
        emb_default = enc.encode(test_texts[0], prompt_template="pcoteol")
        
        # Cast to float32 for safe mixed-precision comparison
        if not torch.allclose(emb_mean.float(), emb_default.float(), atol=1e-3):
            print_success("Override produces different embeddings")
        else:
            print_info("Override similar to default (model-dependent)")
        
        # Test 5: Layer defaults
        print()
        print_info("Test 5: Layer index defaults")
        
        # pcoteol should use layer -2
        emb_l2_default = enc.encode("Test", prompt_template="pcoteol", layer_index=None)
        emb_l2_explicit = enc.encode("Test", prompt_template="pcoteol", layer_index=-2)
        
        # Cast to float32 for safe comparison
        if torch.allclose(emb_l2_default.float(), emb_l2_explicit.float(), atol=1e-6):
            print_success("pcoteol correctly defaults to layer -2")
        else:
            print_error("pcoteol layer default mismatch")
            return False
        
        # mean should use layer -1
        emb_l1_default = enc.encode("Test", pooling_method="mean", layer_index=None)
        emb_l1_explicit = enc.encode("Test", pooling_method="mean", layer_index=-1)
        
        # Cast to float32 for safe comparison
        if torch.allclose(emb_l1_default.float(), emb_l1_explicit.float(), atol=1e-6):
            print_success("mean correctly defaults to layer -1")
        else:
            print_error("mean layer default mismatch")
            return False
        
        print()
        print_success("✅ Transformers backend: ALL TESTS PASSED")
        
        # Cleanup before returning
        del enc
        cleanup_gpu_memory()
        
        return True
        
    except Exception as e:
        print_error(f"Transformers backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vllm_backend(
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
) -> Optional[bool]:
    """
    Test vLLM backend with Smart Defaults.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        True if tests pass, False if tests fail, None if vLLM not available.
    """
    print_header("Step 3: vLLM Backend Test")
    
    # Check if vLLM is installed
    try:
        import vllm
        print_success("vLLM is installed")
        try:
            vllm_version = vllm.__version__
            print_info(f"vLLM version: {vllm_version}")
        except AttributeError:
            print_info("vLLM version: unknown")
    except ImportError:
        print_warning("vLLM is NOT installed - skipping vLLM tests")
        print_info("vLLM is optional. Install with: pip install vllm")
        return None  # Not a failure, just skipped
    
    try:
        import torch
        import llemb
        
        # Critical: Ensure GPU memory is clean before vLLM
        print_info("Cleaning GPU memory before vLLM initialization...")
        cleanup_gpu_memory()
        
        print_info(f"Loading model: {model_name}")
        print_info("Backend: vLLM")
        print_info("This may take longer than Transformers...")
        
        # Initialize encoder with vLLM
        # Use lower memory utilization for compatibility
        enc = llemb.Encoder(
            model_name=model_name,
            backend="vllm",
            device="cuda",
            gpu_memory_utilization=0.5,  # Conservative to avoid OOM
            enforce_eager=True  # More stable
        )
        
        print_success("Model loaded with vLLM backend")
        
        # Test Smart Default API
        print()
        print_info("Testing Smart Default API with vLLM")
        
        test_texts = [
            "vLLM Test",
            "Testing vLLM inference with Smart Defaults"
        ]
        
        # Test 1: Smart Default with pcoteol
        print_info("Test 1: prompt_template='pcoteol' (auto last_token)")
        emb1 = enc.encode(test_texts[0], prompt_template="pcoteol")
        print_success(f"Embedding: shape={emb1.shape}, dtype={emb1.dtype}")
        
        # Test 2: Smart Default with prompteol
        print_info("Test 2: prompt_template='prompteol'")
        emb2 = enc.encode(test_texts[1], prompt_template="prompteol")
        print_success(f"Embedding: shape={emb2.shape}, dtype={emb2.dtype}")
        
        # Test 3: Batch processing
        print_info("Test 3: Batch processing")
        emb_batch = enc.encode(test_texts, prompt_template="ke")
        print_success(f"Batch: shape={emb_batch.shape}, dtype={emb_batch.dtype}")
        
        # Validate embeddings
        if torch.isnan(emb_batch).any():
            print_error("Embeddings contain NaN values")
            return False
        
        if torch.isinf(emb_batch).any():
            print_error("Embeddings contain Inf values")
            return False
        
        print_success("All embeddings are valid")
        
        # Test 4: Explicit override
        print_info("Test 4: Explicit pooling_method override")
        emb_mean = enc.encode(
            test_texts[0],
            pooling_method="mean",
            prompt_template="pcoteol"
        )
        emb_default = enc.encode(test_texts[0], prompt_template="pcoteol")
        
        # Cast to float32 for safe comparison
        if not torch.allclose(emb_mean.float(), emb_default.float(), atol=1e-3):
            print_success("Override produces different embeddings")
        else:
            print_info("Override similar to default (model-dependent)")
        
        print()
        print_success("✅ vLLM backend: ALL TESTS PASSED")
        
        # Cleanup
        del enc
        cleanup_gpu_memory()
        
        return True
        
    except Exception as e:
        print_error(f"vLLM backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """
    Main verification routine.
    
    Returns:
        0 if all required checks pass, 1 otherwise.
    """
    print("\n" + "=" * 80)
    print("  llemb Comprehensive GPU Verification Script (v0.2.2)")
    print("  Testing: Transformers (Quantization) + vLLM + Smart Defaults API")
    print("=" * 80)
    
    results = {
        "environment": False,
        "transformers": False,
        "vllm": None  # None means skipped
    }
    
    # Step 1: Environment check
    if not check_environment():
        print_error("\n❌ VERIFICATION FAILED: Environment not ready")
        return 1
    
    results["environment"] = True
    
    # Step 2: Transformers backend test
    if not test_transformers_backend():
        print_error("\n❌ VERIFICATION FAILED: Transformers backend test failed")
        return 1
    
    results["transformers"] = True
    
    # Step 3: vLLM backend test (optional)
    vllm_result = test_vllm_backend()
    results["vllm"] = vllm_result
    
    if vllm_result is False:  # Failed (not skipped)
        print_error("\n❌ VERIFICATION FAILED: vLLM backend test failed")
        return 1
    
    # Print final summary
    print_header("🎉 VERIFICATION COMPLETE 🎉")
    print()
    print("Test Results Summary:")
    print(f"  {'✓' if results['environment'] else '✗'} Environment & CUDA check")
    print(f"  {'✓' if results['transformers'] else '✗'} Transformers backend (with quantization)")
    
    if results['vllm'] is True:
        print(f"  ✓ vLLM backend")
    elif results['vllm'] is None:
        print(f"  ⊘ vLLM backend (skipped - not installed)")
    else:
        print(f"  ✗ vLLM backend")
    
    print()
    
    if results["transformers"]:
        print("✅ Core functionality verified successfully!")
        print()
        print("Your GPU environment is ready to use llemb v0.2.2!")
        
        if results["vllm"] is None:
            print()
            print("Note: vLLM backend was not tested (not installed).")
            print("Install vLLM for high-throughput inference: pip install vllm")
    else:
        print("❌ Core functionality tests failed.")
        return 1
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
