from src.dataset import (
    available_datasets,
    resolve_dataset_paths,
)

print(available_datasets())

event_path, image_path = resolve_dataset_paths(
    "badminton"
)

print(event_path)

print(image_path)