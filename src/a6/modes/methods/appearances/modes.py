import dataclasses
import datetime
from collections.abc import Iterator

import a6.modes.methods.appearances.mode as _mode


@dataclasses.dataclass
class Modes:
    modes: list[_mode.Mode]

    def __iter__(self) -> Iterator:
        yield from self.modes

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
