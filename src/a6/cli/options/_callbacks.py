from collections.abc import Callable
from typing import Any

import click


def cast_optional(type_: type) -> Callable[[str], Any | None]:
    """Create a function to cast an optional value.

    Returns
    -------
    Callable
        Casts an optional value to the given type.

    """

    def parse(ctx, param: str, value: str) -> type_:
        if value is None or value.lower() == "none":
            return None
        try:
            return type_(value)
        except ValueError:
            raise click.BadParameter(
                f"Invalid value for {param!r}: "
                f"'{value}' is not a valid {type_}."
            )

    return parse
