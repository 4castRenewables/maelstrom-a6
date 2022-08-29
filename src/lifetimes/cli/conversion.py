import typing as t


def string_to_bool(s: str) -> bool:
    """Convert string to respective bool.

    Raises
    ------
    ValueError
        Given string not a valid bool value.

    Notes
    -----
    Phrases that will be converted to `True`.

    """
    valid_true = {"true", "t", "1", "yes", "y"}
    valid_false = {"false", "f", "0", "no", "n"}
    s_lower_case = s.lower()
    if s_lower_case in valid_true:
        return True
    elif s_lower_case in valid_false:
        return False
    else:
        all_valid_booleans = valid_true | valid_false
        raise ValueError(
            f"Given value not valid for boolean. "
            f"Valid booleans: {all_valid_booleans}"
        )


def cast_optional(type_: t.Type) -> t.Callable[[str], t.Optional[t.Any]]:
    """Create a function to cast an optional value.

    Returns a function that allows casting an optional value to the given type.


    """

    def parse(s: str) -> type_:
        if s.lower() == "none":
            return None
        try:
            return type_(s)
        except ValueError:
            raise ValueError(
                f"Invalid argument: Could not convert {s} to type {type_}"
            )

    return parse
