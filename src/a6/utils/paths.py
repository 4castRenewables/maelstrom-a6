import pathlib


def list_files(path: str | pathlib.Path, pattern: str) -> list[pathlib.Path]:
    """List all files in a given path matching the pattern."""
    if isinstance(path, str):
        path = pathlib.Path(path)
    return sorted(f for f in path.glob(pattern) if f.is_file())
