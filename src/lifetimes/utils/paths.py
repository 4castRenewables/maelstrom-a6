import pathlib
from typing import Union


def list_files(
    path: Union[str, pathlib.Path], pattern: str
) -> list[pathlib.Path]:
    """List all files in a given path matching the pattern."""
    if isinstance(path, str):
        path = pathlib.Path(path)
    return sorted(path.glob(pattern))
