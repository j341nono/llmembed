# GPU Verification Script

This directory contains verification scripts for testing llemb in various environments.

## verify_gpu.py

A comprehensive GPU verification script that tests llemb v0.2.2 with quantization and the new Smart Defaults API.

### What It Tests

1. **CUDA Availability**: Verifies that CUDA is available and detects GPU information
2. **bitsandbytes Installation**: Checks that bitsandbytes is properly installed and can link to CUDA
3. **Model Loading with Quantization**: Tests loading a model with 4-bit quantization enabled
4. **Smart Defaults API**: Tests the new v0.2.2 feature where `pooling_method` automatically defaults to `last_token` when using `prompt_template`
5. **Inference Validation**: Verifies that embeddings are generated correctly without NaN or Inf values
6. **Layer Index Defaults**: Confirms that layer indices are correctly defaulted based on template type
7. **Mixed Precision Handling**: All tensor comparisons use `.float()` casting to safely handle Float16/Float32 differences from quantized models

### Prerequisites

- CUDA-capable GPU with NVIDIA drivers installed
- PyTorch with CUDA support
- bitsandbytes (automatically installed with llemb v0.2.2+)

### Installation & Execution

#### Option 1: Install from Source (Development)

If you're working with the development version or have made local changes:

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/j341nono/llemb.git
cd llemb

# Install in editable mode with all dependencies
pip install -e .

# Run the verification script
python scripts/verify_gpu.py

# Or run directly (if executable)
./scripts/verify_gpu.py
```

#### Option 2: Install from PyPI (Production)

If you want to test the published package:

```bash
# Install llemb (bitsandbytes is included as of v0.2.2)
pip install llemb

# Download just the verification script
wget https://raw.githubusercontent.com/j341nono/llemb/main/scripts/verify_gpu.py

# Run the script
python verify_gpu.py
```

#### Option 3: Using uv (Recommended for Development)

```bash
# Sync all dependencies including dev tools
uv sync --all-extras

# Run the script using uv
uv run python scripts/verify_gpu.py
```

### Expected Output

On a successful run, you should see output similar to:

```
================================================================================
  llemb GPU Verification Script (v0.2.2)
  Testing: Quantization + Smart Defaults API
================================================================================

================================================================================
  Step 1: CUDA Availability Check
================================================================================
✓ CUDA is available (version: 12.1)
ℹ Detected 1 CUDA device(s)
ℹ Primary device: NVIDIA GeForce RTX 3090

================================================================================
  Step 2: bitsandbytes Dependency Check
================================================================================
✓ bitsandbytes is installed
ℹ bitsandbytes version: 0.43.0
✓ bitsandbytes can successfully link to CUDA

================================================================================
  Step 3: Model Loading with 4BIT Quantization
================================================================================
ℹ Loading model: HuggingFaceTB/SmolLM2-135M
ℹ Quantization: 4bit
ℹ This may take a moment...
✓ Model loaded successfully with 4bit quantization
ℹ Model is on device: cuda:0

================================================================================
  Step 4: Inference Test with Smart Defaults API
================================================================================
ℹ Testing new Smart Default API (v0.2.2)
ℹ Feature: pooling_method auto-defaults to 'last_token' with prompt_template

ℹ Test 1: Using prompt_template='pcoteol' (should auto-use last_token)
✓ Embedding generated: shape=torch.Size([1, 576]), dtype=torch.float32

ℹ Test 2: Using prompt_template='ke' (should auto-use last_token)
✓ Embedding generated: shape=torch.Size([1, 576]), dtype=torch.float32

ℹ Test 3: Batch processing with Smart Defaults
✓ Batch embeddings: shape=torch.Size([3, 576]), dtype=torch.float32
✓ All embeddings are valid (no NaN or Inf)

ℹ Test 4: Explicit pooling_method='mean' (overrides Smart Default)
✓ Embedding with explicit mean: shape=torch.Size([1, 576])
✓ Explicit override correctly produces different embeddings

================================================================================
  Step 5: Layer Index Defaults Test
================================================================================
ℹ Testing Smart Layer Defaults:
ℹ   - 'pcoteol' and 'ke' templates should use layer -2
ℹ   - 'prompteol' and no template should use layer -1
✓ pcoteol correctly defaults to layer -2
✓ mean pooling correctly defaults to layer -1

================================================================================
  🎉 VERIFICATION SUCCESS 🎉
================================================================================

All checks passed successfully!

Summary:
  ✓ CUDA is available and working
  ✓ bitsandbytes is properly installed
  ✓ Model loading with quantization works
  ✓ Smart Default API (v0.2.2) works correctly
  ✓ Inference produces valid embeddings
  ✓ Layer index defaults work as expected

Your GPU environment is ready to use llemb v0.2.2!
================================================================================
```

### Troubleshooting

#### "CUDA is NOT available"

**Possible causes:**
- No NVIDIA GPU detected
- NVIDIA drivers not installed or outdated
- PyTorch installed without CUDA support

**Solutions:**
```bash
# Check if GPU is detected
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### "bitsandbytes is NOT installed"

**Possible causes:**
- Installation failed
- Version conflict
- CUDA library mismatch

**Solutions:**
```bash
# Reinstall bitsandbytes
pip install --upgrade --force-reinstall bitsandbytes>=0.48.2

# If you get CUDA library errors, try:
pip install bitsandbytes --no-binary bitsandbytes
```

#### "Failed to load model"

**Possible causes:**
- Insufficient GPU memory
- Model not found on HuggingFace Hub
- Network issues

**Solutions:**
```bash
# Try with 8-bit quantization (uses less memory)
# Modify the script to use quantization="8bit"

# Or use an even smaller model
# Modify the script to use model_name="gpt2" (124M parameters)

# Check available GPU memory
nvidia-smi
```

### Customization

You can modify the script to test with different models or quantization settings:

```python
# In verify_gpu.py, modify the test_model_loading function call:

# Use 8-bit quantization instead of 4-bit
enc = test_model_loading(quantization="8bit")

# Use a different model
enc = test_model_loading(
    model_name="meta-llama/Llama-2-7b-hf",  # Requires ~4GB GPU memory with 4-bit
    quantization="4bit"
)
```

### Exit Codes

- `0`: All checks passed successfully
- `1`: One or more checks failed

### Advanced Usage

Run specific tests by importing functions:

```python
from scripts.verify_gpu import check_cuda, check_bitsandbytes

# Just check CUDA availability
if check_cuda():
    print("CUDA is ready!")

# Just check bitsandbytes
if check_bitsandbytes():
    print("bitsandbytes is ready!")
```

## Contributing

If you encounter issues with this verification script or have suggestions for improvement, please open an issue on GitHub.

## Related Documentation

- [llemb README](../README.md)
- [Migration Guide (v0.2.2)](../README.md#migration-guide)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
