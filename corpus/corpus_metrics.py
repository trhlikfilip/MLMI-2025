import argparse, os, math, warnings
from pathlib import Path

import pandas as pd
import spacy
import textdescriptives as td
from tqdm.auto import tqdm

METRIC_SET = [
    "descriptive_stats", "readability", "dependency_distance",
    "pos_proportions", "coherence", "quality", "information_theory",
]

def yield_blocks(path: Path, char_limit: int):
    buf, size = [], 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if size + len(line) + 1 > char_limit and buf:
                yield "".join(buf)
                buf, size = [line], len(line) + 1
            else:
                buf.append(line)
                size += len(line) + 1
        if buf:
            yield "".join(buf)

def every_nth(iterable, n: int):
    return (block for i, block in enumerate(iterable) if i % n == 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--char_limit", type=int, default=512 * 16)
    ap.add_argument("--sample_every", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    warnings.filterwarnings("ignore", message=r"\[W007\]")
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textdescriptives/all")

    estimated_chunks = math.ceil(os.path.getsize(args.input) / args.char_limit)
    estimated_sampled = math.ceil(estimated_chunks / max(1, args.sample_every))

    blocks = yield_blocks(args.input, args.char_limit)
    sampled_blocks = every_nth(blocks, args.sample_every)

    dfs = []
    for doc in tqdm(
        nlp.pipe(sampled_blocks, batch_size=args.batch_size),
        total=estimated_sampled, unit="chunk", desc="Scoring"
    ):
        dfs.append(td.extract_df([doc], metrics=METRIC_SET))

    if not dfs:
        print("No data processed.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    num_cols = df_all.select_dtypes("number").columns
    summary = df_all[num_cols].mean().to_frame().T

    if args.output:
        summary.to_csv(args.output, index=False)
    else:
        print(summary)

if __name__ == "__main__":
    main()
