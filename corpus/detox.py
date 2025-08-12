import os, re, argparse
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm.auto import tqdm
from together import Together

client = Together()
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

_PROMPT = (
    "Below is a list of sentences that have been labelled toxic or hateful. "
    "Rewrite each sentence so it is no longer toxic or hateful while preserving the original meaning. "
    "Return the rewritten sentences in the same order without numbering or comments, one per line, and nothing else:\n\n{}"
)

_num_prefix = re.compile(r"^\s*\d+[.)]\s*")

def _clean(s: str) -> str:
    return " ".join(s.splitlines()).strip()

def _rewrite(sentences: List[str], temperature: float = 0.0, model: str = MODEL) -> List[str]:
    body = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = _PROMPT.format(body)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    raw = [l.strip() for l in resp.choices[0].message.content.splitlines() if l.strip()]
    cleaned = [_num_prefix.sub("", l) for l in raw]
    if len(cleaned) < len(sentences):
        cleaned.extend([""] * (len(sentences) - len(cleaned)))
    elif len(cleaned) > len(sentences):
        cleaned = cleaned[: len(sentences)]
    return cleaned

def detoxify_df(
    df: pd.DataFrame,
    sentence_col: str = "sentence",
    hate_col: str = "hate",
    tox_col: str = "toxicity",
    *,
    threshold: float = 0.5,
    batch_size: int = 8,
    max_workers: int = 4,
    temperature: float = 0.0,
    model: str = MODEL,
    outfile: Optional[str] = None
) -> List[str]:
    bad = df[(df[hate_col] > threshold) | (df[tox_col] > threshold)][sentence_col].astype(str).tolist()
    if not bad:
        return []
    clean_sentences = [_clean(s) for s in bad]
    batches = [clean_sentences[i: i + batch_size] for i in range(0, len(clean_sentences), batch_size)]
    results = [None] * len(bad)
    next_to_flush = 0
    if outfile:
        open(outfile, "w", encoding="utf-8").close()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_slice = {
            pool.submit(_rewrite, batch, temperature, model): (start, len(batch))
            for start, batch in ((i * batch_size, b) for i, b in enumerate(batches))
        }
        for fut in tqdm(as_completed(fut_to_slice), total=len(fut_to_slice), desc="Detoxifying", unit="batch"):
            start, n = fut_to_slice[fut]
            try:
                results[start: start + n] = fut.result()
            except:
                results[start: start + n] = [""] * n
            if outfile:
                while next_to_flush < len(results) and results[next_to_flush] is not None:
                    new = results[next_to_flush]
                    if new:
                        orig = bad[next_to_flush].replace("\t", " ").replace("\n", " ").replace("\r", "")
                        with open(outfile, "a", encoding="utf-8") as fh:
                            fh.write(f"{orig},-------- {new}\n")
                    next_to_flush += 1
    return [r if r is not None else "" for r in results]

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_csv", required=False)
    ap.add_argument("--outfile_pairs", required=False)
    ap.add_argument("--sentence_col", default="sentence")
    ap.add_argument("--hate_col", default="hate")
    ap.add_argument("--tox_col", default="toxicity")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    cleaned = detoxify_df(
        df,
        sentence_col=args.sentence_col,
        hate_col=args.hate_col,
        tox_col=args.tox_col,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        temperature=args.temperature,
        model=args.model,
        outfile=args.outfile_pairs,
    )

    if args.out_csv:
        bad_mask = (df[args.hate_col] > args.threshold) | (df[args.tox_col] > args.threshold)
        bad_df = df.loc[bad_mask, [args.sentence_col]].copy()
        bad_df["cleaned"] = cleaned
        bad_df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
