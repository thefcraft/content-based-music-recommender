import os
from typing import Any, Literal, Annotated, overload


@overload
def get_files_in_directory(
    path: os.PathLike[str], *, return_filename: Literal[False] = False
) -> list[str]: ...
@overload
def get_files_in_directory(
    path: os.PathLike[str], *, return_filename: Literal[True]
) -> Annotated[list[tuple[str, str]], "list of (filename, filepath)"]: ...
def get_files_in_directory(
    path: os.PathLike[str], *, return_filename: bool = False
) -> list[str] | list[tuple[str, str]]:
    """Get all files in the specified directory."""
    if not path:
        raise ValueError("Path cannot be empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} does not exist.")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    if return_filename:
        return [
            (filename, filepath)
            for filename in os.listdir(path)
            if os.path.isfile(filepath := os.path.join(path, filename))
        ]
    return [
        filepath
        for filename in os.listdir(path)
        if os.path.isfile(filepath := os.path.join(path, filename))
    ]


def cast[T](val: Any, _typ: type[T], _annotation: str | None = None) -> T:
    """same as typing.cast but first argument is val and second argument is type."""
    return val


def annotate[T](val: T, _: str | None = None, **kwargs: str) -> T:
    """just for any annotation."""
    return val
