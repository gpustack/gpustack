from typing import List, Literal, Optional, Union

from openai.types.embedding import Embedding
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types import Completion
from openai.types.completion_choice import CompletionChoice


class EmbeddingExt(Embedding):
    embedding: Union[str, List[float]]
    """The embedding vector generated by the model. Supported for base64 encoded and float list formats."""


class CreateEmbeddingResponseExt(CreateEmbeddingResponse):
    data: List[EmbeddingExt]
    """The list of embeddings generated by the model."""


class CompletionChoiceExt(CompletionChoice):
    """Extended CompletionChoice model to include nullable finish_reason."""

    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None
    """The reason the model stopped generating tokens."""


class CompletionExt(Completion):
    choices: List[CompletionChoiceExt]
    """The list of completion choices the model generated for the input prompt."""
