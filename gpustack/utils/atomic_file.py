from pathlib import Path
from tempfile import NamedTemporaryFile


def write_text_atomically(path: Path, content: str) -> None:
    """Replace a UTF-8 file atomically using a private temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        temporary_path = Path(output.name)
        try:
            output.write(content)
            output.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
