import abc
from typing import Any, List, Optional, Union


class Backend(abc.ABC):
    @abc.abstractmethod
    def encode(
        self,
        text: Union[str, List[str]],
        pooling_method: Optional[str] = None,
        layer_index: Optional[int] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Encode text into embeddings.

        Args:
            text: Input text or list of texts.
            pooling_method: Pooling method ('mean', 'last_token', 'eos_token').
                          If None, defaults to 'last_token' with template, otherwise 'mean'.
            layer_index: Layer index to extract embeddings from.
            prompt_template: Optional prompt template ('prompteol', 'pcoteol', 'ke').
            **kwargs: Additional backend-specific arguments.

        Returns:
            Embeddings as numpy array or torch tensor.
        """
        pass
