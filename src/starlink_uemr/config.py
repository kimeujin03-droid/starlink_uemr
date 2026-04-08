from pathlib import Path
import yaml


def load_yaml(path: str) -> dict:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path_obj, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # 과학표기법 문자열을 float로 변환
    return _convert_floats(data)


def _convert_floats(obj):
    """Recursively convert string floats to float type"""
    if isinstance(obj, dict):
        return {k: _convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats(v) for v in obj]
    elif isinstance(obj, str):
        try:
            return float(obj)
        except (ValueError, TypeError):
            return obj
    else:
        return obj


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
