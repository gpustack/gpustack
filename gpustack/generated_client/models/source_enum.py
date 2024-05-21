from enum import Enum


class SourceEnum(str, Enum):
    HUGGINGFACE = "huggingface"
    S3 = "s3"

    def __str__(self) -> str:
        return str(self.value)
