import yaml
import torch
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device(config) -> torch.device:
    device_config = config.base.device

    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        return torch.device("cuda")

    if device_config == "cpu":
        return torch.device("cpu")

    if device_config == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    raise ValueError(f"Unsupported device: {device_config}")

def setup_seed(config) -> None:
    seed = config.base.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)