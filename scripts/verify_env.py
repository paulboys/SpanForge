import pathlib
import sys

import torch

# Ensure project root is on sys.path so 'src' package resolves even if run from a shell
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config


def main():
    config = get_config()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (if any): {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("Running CPU-only inference.")
    print(f"Model name: {config.model_name}")
    print(f"Device selected by config: {config.device}")


if __name__ == "__main__":
    main()
