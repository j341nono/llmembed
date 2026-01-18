import abc
from typing import Any, List, Optional, Union


class Backend(abc.ABC):
    @abc.abstractmethod
    def encode(
        self,
        text: Union[str, List[str]],
        pooling: str = "mean",
        layer_index: Optional[int] = None,
        **kwargs: Any
    ) -> Any:
        """
        Encode text into embeddings.

        Args:
            text: Input text or list of texts.
            pooling: Pooling strategy.
            layer_index: Layer index to extract embeddings from.
            **kwargs: Additional backend-specific arguments.

        Returns:
            Embeddings as numpy array or torch tensor.
        """
        pass
