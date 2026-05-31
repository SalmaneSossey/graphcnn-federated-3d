"""Top-level smoke entrypoint for the graphcnn-federated-3d project."""

from __future__ import annotations

from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/data.yaml")
    print("Project scaffold is ready.")
    print(f"Default num_points: {config['num_points']}")
    print(f"Default seed: {config['seed']}")


if __name__ == "__main__":
    main()

