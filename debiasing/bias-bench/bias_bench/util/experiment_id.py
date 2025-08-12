import os
import re

#Code optimised for the new models
def _safe(text: str) -> str:

    if not isinstance(text, str):
        text = str(text)
        
    text = text.replace(os.sep, "-").replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")

    return text


def generate_experiment_id(
    name: str,
    model: str | None = None,
    model_name_or_path: str | None = None,
    bias_type: str | None = None,
    seed: int | None = None,
    train_fraction: float | None = None,
) -> str:
    parts: list[str] = [_safe(name)]

    if isinstance(model, str):
        parts.append(f"m-{_safe(model)}")
    if isinstance(model_name_or_path, str):
        parts.append(f"c-{_safe(model_name_or_path)}")
    if isinstance(bias_type, str):
        parts.append(f"t-{_safe(bias_type)}")
    if isinstance(seed, int):
        parts.append(f"s-{seed}")
    if isinstance(train_fraction, float):
        parts.append(f"f-{train_fraction}")

    return "_".join(parts)
