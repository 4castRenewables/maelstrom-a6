import pathlib


def list_files(
    path: str | pathlib.Path, pattern: str, recursive: bool = False
) -> list[pathlib.Path]:
    """List all files in a given path matching the pattern."""
    if isinstance(path, str):
        path = pathlib.Path(path)
    results = path.rglob(pattern) if recursive else path.glob(pattern)
    return sorted(f for f in results if f.is_file())
