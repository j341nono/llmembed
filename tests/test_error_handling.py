"""
Test error handling and input validation in the llemb library.

These tests use unittest.mock to simulate backend behavior without requiring GPU access.
"""
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from llemb.core import VALID_POOLING_METHODS, Encoder


class TestCoreValidation:
    """Test input validation in core.Encoder.encode()"""

    @patch("llemb.core.TransformersBackend")
    def test_invalid_pooling_method_raises_error(self, mock_backend_class):
        """Test that invalid pooling_method raises ValueError"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        # Test various invalid pooling methods
        invalid_methods = ["invalid", "average", "max", "min", "first", ""]

        for invalid_method in invalid_methods:
            with pytest.raises(
                ValueError, match=f"Invalid pooling_method: '{invalid_method}'"
            ):
                encoder.encode(
                    text="test text", pooling_method=invalid_method, batch_size=None
                )

    @patch("llemb.core.TransformersBackend")
    def test_valid_pooling_methods_accepted(self, mock_backend_class):
        """Test that all valid pooling methods are accepted"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_instance.encode.return_value = torch.zeros(1, 768)
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        # Test all valid pooling methods
        for valid_method in VALID_POOLING_METHODS:
            result = encoder.encode(
                text="test text", pooling_method=valid_method, batch_size=None
            )
            assert result is not None
            mock_backend_instance.encode.assert_called()

    @patch("llemb.core.TransformersBackend")
    def test_zero_batch_size_raises_error(self, mock_backend_class):
        """Test that batch_size=0 raises ValueError"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        with pytest.raises(
            ValueError, match="batch_size must be a positive integer, got: 0"
        ):
            encoder.encode(text="test text", batch_size=0)

    @patch("llemb.core.TransformersBackend")
    def test_negative_batch_size_raises_error(self, mock_backend_class):
        """Test that negative batch_size raises ValueError"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        with pytest.raises(
            ValueError, match="batch_size must be a positive integer, got: -1"
        ):
            encoder.encode(text="test text", batch_size=-1)

        with pytest.raises(
            ValueError, match="batch_size must be a positive integer, got: -10"
        ):
            encoder.encode(text="test text", batch_size=-10)

    @patch("llemb.core.TransformersBackend")
    def test_positive_batch_size_accepted(self, mock_backend_class):
        """Test that positive batch_size values are accepted"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_instance.encode.return_value = torch.zeros(2, 768)
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        # Test various positive batch sizes
        valid_batch_sizes = [1, 2, 8, 16, 32, 100]

        for batch_size in valid_batch_sizes:
            result = encoder.encode(
                text=["text1", "text2"], batch_size=batch_size, pooling_method="mean"
            )
            assert result is not None

    @patch("llemb.core.TransformersBackend")
    def test_none_batch_size_accepted(self, mock_backend_class):
        """Test that batch_size=None (default) is accepted"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_instance.encode.return_value = torch.zeros(1, 768)
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        # Should not raise error
        result = encoder.encode(text="test text", batch_size=None)
        assert result is not None


class TestVLLMBackendLayerValidation:
    """Test layer_index validation in vLLM backend"""

    @patch("llemb.core.VLLMBackend")
    def test_vllm_unsupported_layer_index_raises_error(self, mock_vllm_class):
        """Test that vLLM backend raises ValueError for unsupported layer_index"""
        # Setup mock backend that raises ValueError for non-last-layer requests
        mock_backend_instance = Mock()

        def mock_encode(*args, **kwargs):
            layer_index = kwargs.get("layer_index")
            if layer_index is not None and layer_index != -1:
                raise ValueError(
                    f"layer_index={layer_index} is not supported by vLLM backend. "
                    "vLLM currently only supports the last layer (layer_index=-1)."
                )
            return torch.zeros(1, 768)

        mock_backend_instance.encode.side_effect = mock_encode
        mock_vllm_class.return_value = mock_backend_instance

        # Create encoder with vLLM backend
        with patch("llemb.core.VLLMBackend", mock_vllm_class):
            encoder = Encoder(model_name="test-model", backend="vllm")

        # Test that non-last-layer indices raise errors
        invalid_layer_indices = [-2, -3, 0, 1, 5, 10]

        for layer_idx in invalid_layer_indices:
            with pytest.raises(
                ValueError,
                match=f"layer_index={layer_idx} is not supported by vLLM backend",
            ):
                encoder.encode(text="test text", layer_index=layer_idx)

    @patch("llemb.core.VLLMBackend")
    def test_vllm_last_layer_accepted(self, mock_vllm_class):
        """Test that vLLM backend accepts layer_index=-1"""
        # Setup mock backend
        mock_backend_instance = Mock()

        def mock_encode(*args, **kwargs):
            layer_index = kwargs.get("layer_index")
            if layer_index is not None and layer_index != -1:
                raise ValueError(
                    f"layer_index={layer_index} is not supported by vLLM backend."
                )
            return torch.zeros(1, 768)

        mock_backend_instance.encode.side_effect = mock_encode
        mock_vllm_class.return_value = mock_backend_instance

        # Create encoder with vLLM backend
        with patch("llemb.core.VLLMBackend", mock_vllm_class):
            encoder = Encoder(model_name="test-model", backend="vllm")

        # Should not raise error for layer_index=-1
        result = encoder.encode(text="test text", layer_index=-1)
        assert result is not None

    @patch("llemb.core.VLLMBackend")
    def test_vllm_none_layer_index_accepted(self, mock_vllm_class):
        """Test that vLLM backend accepts layer_index=None (default)"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_instance.encode.return_value = torch.zeros(1, 768)
        mock_vllm_class.return_value = mock_backend_instance

        # Create encoder with vLLM backend
        with patch("llemb.core.VLLMBackend", mock_vllm_class):
            encoder = Encoder(model_name="test-model", backend="vllm")

        # Should not raise error for layer_index=None
        result = encoder.encode(text="test text", layer_index=None)
        assert result is not None


class TestCombinedValidation:
    """Test combinations of validation scenarios"""

    @patch("llemb.core.TransformersBackend")
    def test_multiple_invalid_params_first_error_raised(self, mock_backend_class):
        """Test that when multiple params are invalid, the first validation error is raised"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        # Both pooling_method and batch_size are invalid
        # Should raise pooling_method error first (as it's validated first)
        with pytest.raises(ValueError, match="Invalid pooling_method"):
            encoder.encode(text="test text", pooling_method="invalid", batch_size=-1)

    @patch("llemb.core.TransformersBackend")
    def test_valid_params_with_smart_defaults(self, mock_backend_class):
        """Test that smart defaults work correctly with validation"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_instance.encode.return_value = torch.zeros(1, 768)
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        # Test smart default: prompt_template provided -> pooling_method="last_token"
        result = encoder.encode(
            text="test text", prompt_template="prompteol", pooling_method=None
        )
        assert result is not None

        # Verify that the backend was called with the correct pooling_method
        call_kwargs = mock_backend_instance.encode.call_args[1]
        assert call_kwargs["pooling_method"] == "last_token"

    @patch("llemb.core.TransformersBackend")
    def test_explicit_valid_params_override_defaults(self, mock_backend_class):
        """Test that explicit valid parameters override smart defaults"""
        # Setup mock backend
        mock_backend_instance = Mock()
        mock_backend_instance.encode.return_value = torch.zeros(1, 768)
        mock_backend_class.return_value = mock_backend_instance

        # Create encoder
        encoder = Encoder(model_name="test-model", backend="transformers")

        # Explicitly set pooling_method to "mean" even with prompt_template
        result = encoder.encode(
            text="test text", prompt_template="prompteol", pooling_method="mean"
        )
        assert result is not None

        # Verify that the backend was called with explicit pooling_method
        call_kwargs = mock_backend_instance.encode.call_args[1]
        assert call_kwargs["pooling_method"] == "mean"


class TestBackendInitialization:
    """Test backend initialization errors"""

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend name raises ValueError"""
        with pytest.raises(ValueError, match="Unknown backend: invalid_backend"):
            Encoder(model_name="test-model", backend="invalid_backend")

    @patch("llemb.core.TransformersBackend")
    def test_valid_transformers_backend(self, mock_backend_class):
        """Test that transformers backend initializes successfully"""
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance

        encoder = Encoder(model_name="test-model", backend="transformers")
        assert encoder.backend_name == "transformers"
        assert encoder.backend_instance == mock_backend_instance

    @patch("llemb.core.VLLMBackend")
    def test_valid_vllm_backend(self, mock_vllm_class):
        """Test that vllm backend initializes successfully"""
        mock_backend_instance = Mock()
        mock_vllm_class.return_value = mock_backend_instance

        with patch("llemb.core.VLLMBackend", mock_vllm_class):
            encoder = Encoder(model_name="test-model", backend="vllm")
            assert encoder.backend_name == "vllm"
            assert encoder.backend_instance == mock_backend_instance
