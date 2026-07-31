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

@dataclass
class EventFrame:
    """
    Event accumulation over a time interval.
    """

    image: np.ndarray
    t_start: float
    t_end: float

@dataclass
class EventGradient:
    """
    Spatial gradient of the event prior.
    """

    grad_x: np.ndarray
    grad_y: np.ndarray
    magnitude: np.ndarray    

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
# Filtering and Accumulation
# ==========================================================

def filter_events_by_time(
    events: EventStream,
    t_start: float,
    t_end: float,
) -> EventStream:
    """
    Return events within [t_start, t_end].
    """

    mask = (events.t >= t_start) & (events.t <= t_end)

    return EventStream(
        x=events.x[mask],
        y=events.y[mask],
        t=events.t[mask],
        p=events.p[mask],
    )

def accumulate_events(
    events: EventStream,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Equation (5)

    Accumulate polarity values into an image.
    """

    H, W = image_shape

    frame = np.zeros((H, W), dtype=np.float32)

    np.add.at(
        frame,
        (events.y, events.x),
        events.p,
    )

    return frame

def compute_event_prior(
    events: EventStream,
    tau: float,
    image_shape: tuple[int, int],
) -> EventFrame:
    """
    Compute I_tau(t).

    Parameters
    ----------
    tau
        Time window.

    image_shape
        (height, width)
    """

    t_end = events.t[-1]

    t_start = t_end - tau

    window_events = filter_events_by_time(
        events,
        t_start,
        t_end,
    )

    prior = accumulate_events(
        window_events,
        image_shape,
    )

    return EventFrame(
        image=prior,
        t_start=t_start,
        t_end=t_end,
    )

# ==========================================================
#   Gradient Computation 
# ==========================================================
import cv2

def compute_event_gradient(
    event_frame: EventFrame,
) -> EventGradient:
    """
    Compute spatial gradients of the event prior.
    """

    image = event_frame.image.astype(np.float32)

    grad_x = cv2.Sobel(
        image,
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=3,
    )

    grad_y = cv2.Sobel(
        image,
        cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=3,
    )

    magnitude = np.sqrt(
        grad_x**2 +
        grad_y**2
    )

    return EventGradient(
        grad_x=grad_x,
        grad_y=grad_y,
        magnitude=magnitude,
    )

# ==========================================================
# Visualization
# ==========================================================

import matplotlib.pyplot as plt

def visualize_event_prior(event_frame: EventFrame):

    plt.figure(figsize=(8, 6))

    plt.imshow(
        event_frame.image,
        cmap="seismic",
        vmin=-5,
        vmax=5,
    )

    plt.title("Event Prior")

    plt.colorbar()

    plt.tight_layout()

    plt.show()

def visualize_event_gradient(
    gradient: EventGradient,
):
    """
    Display gradient magnitude.
    """

    plt.figure(figsize=(8,6))

    plt.imshow(
        gradient.magnitude,
        cmap="inferno",
    )

    plt.title("Event Gradient Magnitude")

    plt.colorbar()

    plt.tight_layout()

    plt.show()


# ==========================================================
# Save output
# ==========================================================
from pathlib import Path

def save_event_prior(
    event_frame: EventFrame,
    filename: str,
):
    """
    Save event prior image.
    """

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    output_dir = PROJECT_ROOT / "results"

    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8,6))

    plt.imshow(
        event_frame.image,
        cmap="gray",
    )

    plt.axis("off")

    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def save_event_gradient(
    gradient: EventGradient,
    filename: str,
):
    """
    Save event gradient magnitude image.
    """

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    output_dir = PROJECT_ROOT / "results"

    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8,6))

    plt.imshow(
        gradient.magnitude,
        cmap="inferno",
    )

    plt.axis("off")

    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()    
    


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    filepath = PROJECT_ROOT / "data" / "events" / "badminton.mat"

    events = load_events(filepath)

    validate_events(events)

    print_event_statistics(events)

    event_prior = compute_event_prior(
        events,
        tau=6000,
        image_shape=(260,346),
    )

    gradient = compute_event_gradient(
        event_prior,
    )

    save_event_prior(
        event_prior,
        "event_prior.png",
    )

    save_event_gradient(
        gradient,
        "event_gradient.png",
    )

    visualize_event_prior(event_prior)

    visualize_event_gradient(gradient)