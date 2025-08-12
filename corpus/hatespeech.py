from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
from tqdm.auto import tqdm

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ModuleNotFoundError as exc:
    raise SystemExit("Install required packages: pip install torch transformers tqdm pandas") from exc

TXT_FILE: Path | str = "babyLM_corpus.txt"
CSV_FILE: Path | str | None = 'BabyLM_hate.csv'
DEVICE: str = "auto"
BATCH_SIZE: int = 64
MODEL_NAME: str = "cardiffnlp/twitter-roberta-base-hate-latest"
MAX_LENGTH: int = 256

def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
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

def hate_to_csv(
    txt_file: str | Path,
    csv_file: str | Path | None = None,
    *,
    tokenizer=None,
    model=None,
    batch_size: int | None = None,
    device: str | None = None,
    desc: str = "Scoring hate-speech",
) -> pd.DataFrame:
    csv_path = Path(csv_file) if csv_file else Path(txt_file).with_suffix(".csv")
    sentences = _load_sentences(txt_file)
    dev = _resolve_device(device or DEVICE)
    tok = tokenizer or AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = model or AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(dev).eval()
    bs = batch_size or BATCH_SIZE
    hate_idx = mdl.config.label2id.get("HATE", 1)
    all_scores: list[float] = []
    for batch in tqdm(_batched(sentences, bs), total=(len(sentences) + bs - 1) // bs, desc=desc):
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(dev)
        with torch.no_grad():
            logits = mdl(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[:, hate_idx]
        all_scores.extend(probs.cpu().tolist())
    df = pd.DataFrame({"sentence": sentences, "hate": all_scores})
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved {len(df)} rows to {csv_path} | Avg score: {df['hate'].mean():.4f}")
    return df

if __name__ == "__main__":
    hate_to_csv(TXT_FILE, CSV_FILE)
