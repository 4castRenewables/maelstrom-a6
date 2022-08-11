import pathlib
import typing as t


def list_files(
    path: t.Union[str, pathlib.Path], pattern: str
) -> list[pathlib.Path]:
    """List all files in a given path matching the pattern."""
    if isinstance(path, str):
        path = pathlib.Path(path)
    return sorted(path.glob(pattern))
