from __future__ import annotations
import re
from pathlib import Path
from typing import Sequence
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TXT_FILE: Path | str = "babyLM_corpus.txt"
CSV_FILE: Path | str | None = "BabyLM_sentiment.csv"
DEVICE: str = "auto"
BATCH_SIZE: int = 64
MODEL_NAME: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
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

def sentiment_to_csv(
    txt_file: str | Path,
    csv_file: str | Path | None = None,
    *,
    tokenizer=None,
    model=None,
    batch_size: int | None = None,
    device: str | None = None,
    desc: str = "Scoring sentiment",
) -> pd.DataFrame:
    csv_path = Path(csv_file) if csv_file else Path(txt_file).with_suffix(".csv")
    sentences = _load_sentences(txt_file)
    dev = _resolve_device(device or DEVICE)
    tok = tokenizer or AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = model or AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(dev).eval()
    bs = batch_size or BATCH_SIZE
    id2label = mdl.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    probs_acc = {lbl: [] for lbl in labels}
    pred_ids = []
    for batch in tqdm(_batched(sentences, bs), total=(len(sentences) + bs - 1) // bs, desc=desc):
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(dev)
        with torch.no_grad():
            logits = mdl(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        for i, lbl in enumerate(labels):
            probs_acc[lbl].extend(probs[:, i].cpu().tolist())
        pred_ids.extend(torch.argmax(probs, dim=-1).cpu().tolist())
    pred_labels = [id2label[i] for i in pred_ids]
    data = {"sentence": sentences}
    for lbl in labels:
        data[lbl.lower()] = probs_acc[lbl]
    data["label"] = pred_labels
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    avg = ", ".join(f"{lbl.lower()}={pd.Series(probs_acc[lbl]).mean():.3f}" for lbl in labels)
    print(f"Saved {len(df)} rows to {csv_path} | Avg: {avg}")
    return df

if __name__ == "__main__":
    sentiment_to_csv(TXT_FILE, CSV_FILE)
