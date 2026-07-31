from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# ==========================================================
# Data Structure
# ==========================================================

@dataclass
class EventStream:
    """
    Container for neuromorphic events.

    Attributes
    ----------
    x : np.ndarray
        Horizontal pixel coordinates.

    y : np.ndarray
        Vertical pixel coordinates.

    t : np.ndarray
        Event timestamps.

    p : np.ndarray
        Event polarity (-1 or +1).
    """

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray


# ==========================================================
# Loading
# ==========================================================

def load_events(filepath: str | Path) -> EventStream:
    """
    Load events from MATLAB (.mat) file.

    Expected format
    ---------------
    events : Nx4 matrix

    Columns
    -------
    [timestamp, x, y, polarity]

    where polarity is stored as
        0 -> negative
        1 -> positive

    Returns
    -------
    EventStream
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found:\n{filepath}")

    data = loadmat(filepath)

    if "events" not in data:
        raise KeyError("'events' variable not found inside MAT file.")

    events = data["events"]

    if events.shape[1] != 4:
        raise ValueError(
            f"Expected Nx4 events matrix. Got {events.shape}"
        )

    t = events[:, 0].astype(np.float64)
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)

    # Convert polarity
    p = np.where(events[:, 3] == 0, -1, 1).astype(np.int8)

    return EventStream(
        x=x,
        y=y,
        t=t,
        p=p,
    )


# ==========================================================
# Validation
# ==========================================================

def validate_events(events: EventStream) -> None:
    """
    Validate loaded event stream.
    """

    n = len(events.t)

    if not (
        len(events.x)
        == len(events.y)
        == len(events.t)
        == len(events.p)
    ):
        raise ValueError("Event arrays have inconsistent lengths.")

    if not np.all(np.diff(events.t) >= 0):
        raise ValueError("Event timestamps are not sorted.")

    if not np.all(np.isin(events.p, [-1, 1])):
        raise ValueError("Invalid polarity values.")

    if np.min(events.x) < 0:
        raise ValueError("Negative x coordinate detected.")

    if np.min(events.y) < 0:
        raise ValueError("Negative y coordinate detected.")

    print(f"✓ Event stream validated ({n:,} events)")


# ==========================================================
# Statistics
# ==========================================================

def print_event_statistics(events: EventStream) -> None:
    """
    Display useful event statistics.
    """

    positive = np.sum(events.p == 1)
    negative = np.sum(events.p == -1)

    print("\n" + "=" * 45)
    print("EVENT STREAM STATISTICS")
    print("=" * 45)

    print(f"Total Events      : {len(events.t):,}")
    print(f"Positive Events   : {positive:,}")
    print(f"Negative Events   : {negative:,}")

    print()

    print(f"Time Start        : {events.t[0]}")
    print(f"Time End          : {events.t[-1]}")
    print(f"Duration          : {events.t[-1]-events.t[0]}")

    print()

    print(f"X Range           : {events.x.min()} → {events.x.max()}")
    print(f"Y Range           : {events.y.min()} → {events.y.max()}")

    print("=" * 45)


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    filepath = PROJECT_ROOT / "data" / "events" / "badminton.mat"

    events = load_events(filepath)

    validate_events(events)

    print_event_statistics(events)