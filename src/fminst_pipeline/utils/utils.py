import yaml

def load_yaml(path: str) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)