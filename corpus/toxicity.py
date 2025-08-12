from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from detoxify import Detoxify
from tqdm.auto import tqdm

try:
    import torch
except ModuleNotFoundError:
    torch = None

TXT_FILE: Path | str = "BabyLM_corpus.txt"
CSV_FILE: Path | str | None = 'BabyLM_toxicity.csv'
DEVICE: str = "auto"
BATCH_SIZE: int = 64

def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device in {"cuda", "cpu"}:
        return device
    raise ValueError("DEVICE must be 'auto', 'cuda' or 'cpu'")

def _load_sentences(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _batched(iterable: Sequence[str], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]

def toxicity_to_csv(
    txt_file: str | Path,
    csv_file: str | Path | None = None,
    *,
    model: Optional[Detoxify] = None,
    batch_size: int | None = None,
    device: str | None = None,
    desc: str = "Scoring toxicity",
) -> pd.DataFrame:
    csv_path = Path(csv_file) if csv_file else Path(txt_file).with_suffix(".csv")
    sentences = _load_sentences(txt_file)
    dev = _resolve_device(device or DEVICE)
    mdl = model or Detoxify("original", device=dev)
    bs = batch_size or BATCH_SIZE
    all_scores = []
    for batch in tqdm(_batched(sentences, bs), total=(len(sentences) + bs - 1) // bs, desc=desc):
        preds = mdl.predict(batch)["toxicity"]
        all_scores.extend(preds)
    df = pd.DataFrame({"sentence": sentences, "toxicity": all_scores})
    df.to_csv(csv_path, index=False, float_format="%.4f")
    return df

if __name__ == "__main__":
    toxicity_to_csv(TXT_FILE, CSV_FILE)
