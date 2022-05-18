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
