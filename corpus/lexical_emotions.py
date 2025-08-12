from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List

import pandas as pd
from nrclex import NRCLex

EMOTION_ORDER: List[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

_TOKEN_RE = re.compile(r"\b\w+\b")

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

class EmotionCalculator:
    def __init__(self, tokens: List[str]):
        self.tokens = list(tokens)
        self._cumulative: Dict[str, float] = defaultdict(float)
        self._all: Dict[str, float] = {
            "fear": 0.0,
            "anger": 0.0,
            "anticip": 0.0,
            "trust": 0.0,
            "surprise": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "joy": 0.0,
        }

    def calculate(self) -> List[float]:
        if not self.tokens:
            return [0.0] * len(EMOTION_ORDER)
        for token in self.tokens:
            emo = NRCLex(token)
            for emotion, score in emo.raw_emotion_scores.items():
                self._cumulative[emotion] += score
        total = len(self.tokens)
        for emo, cumul in self._cumulative.items():
            self._all[emo] = cumul / total
        return [self._all.get(e, 0.0) for e in EMOTION_ORDER]

def analyse_text(text: str) -> List[float]:
    return EmotionCalculator(tokenize(text)).calculate()

def _print_corpus_summary(df: pd.DataFrame) -> None:
    means = df[EMOTION_ORDER].mean()
    print("\nCorpus Emotion Averages")
    for emo, val in means.items():
        print(f"{emo.title():<8}: {val:.4f}")

def process_corpus(
    txt_path: str,
    output_csv: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(path)
    records: List[Dict[str, float | str]] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            vec = analyse_text(line)
            records.append({"line": idx, **dict(zip(EMOTION_ORDER, vec)), "text": line.strip()})
    df = pd.DataFrame.from_records(records, columns=["line", *EMOTION_ORDER, "text"])
    df.to_csv(output_csv, index=False)
    if verbose:
        _print_corpus_summary(df)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("emotions.txt")
    parser.add_argument("output_csv")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    process_corpus(args.txt_path, args.output_csv, verbose=not args.no_summary)
