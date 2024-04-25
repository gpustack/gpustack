from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = Field(default=False, description="Whether to stream the response")


class Choice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[Message] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    model: str
    choices: List[Choice]
    created: int


class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
