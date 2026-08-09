"""
dataset.py

Dataset utilities.

Responsible only for:

1. Listing available datasets.
2. Resolving dataset paths.
3. Creating result directories.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVENT_DIR = PROJECT_ROOT / "data" / "events"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
RESULT_DIR = PROJECT_ROOT / "results"


def available_datasets() -> list[str]:
    """
    Return all available dataset names.

    Example
    -------
    ['badminton',
     'keyboard',
     'toy',
     ...]
    """

    return sorted(
        file.stem
        for file in EVENT_DIR.glob("*.mat")
    )


def resolve_dataset_paths(
    dataset_name: str,
) -> tuple[Path, Path]:
    """
    Resolve dataset file paths.

    Parameters
    ----------
    dataset_name

        badminton
        keyboard
        toy
        ...

    Returns
    -------
    event_path
    image_path
    """

    event_path = EVENT_DIR / f"{dataset_name}.mat"
    image_path = IMAGE_DIR / f"{dataset_name}.png"

    if not event_path.exists():
        raise FileNotFoundError(
            f"Event file not found:\n{event_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found:\n{image_path}"
        )

    return event_path, image_path


def get_result_directory(
    dataset_name: str,
) -> Path:
    """
    Create and return

    results/<dataset_name>/
    """

    output_dir = RESULT_DIR / dataset_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir