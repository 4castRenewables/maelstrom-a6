import dataclasses
import datetime
from collections.abc import Iterator

import a6.modes.methods.appearances.mode as _mode


@dataclasses.dataclass
class Modes:
    modes: list[_mode.Mode]

    def __iter__(self) -> Iterator:
        yield from self.modes

    def __getitem__(self, key: int | datetime.datetime) -> _mode.Mode:
        if isinstance(key, int):
            return self.get_mode(label=key)
        elif isinstance(key, datetime.datetime):
            return self.get_appearance(date=key)
        raise KeyError(
            f"Type {type(key)} not supported as key, "
            "must be int or datetime.datetime"
        )

    @property
    def size(self) -> int:
        """Return the number of existing modes."""
        return len(self.modes)

    @property
    def labels(self) -> Iterator[int]:
        """Return all labels."""
        return (mode.label for mode in self.modes)

    def get_mode(self, label: int) -> _mode.Mode:
        """Get a certain mode via label."""
        for mode in self.modes:
            if label == mode.label:
                return mode
        raise ValueError(f"Mode with label {label} not found")

    def get_appearance(self, date: datetime.datetime) -> _mode.Appearance:
        """Return the appearance of a mode for the respective date."""
        for mode in self.modes:
            for appearance in mode.appearances:
                if appearance.start <= date <= appearance.end:
                    return appearance
        raise ValueError(f"No appearance found for {date}")
