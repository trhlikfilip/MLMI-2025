from __future__ import annotations
import argparse
import csv
import gc
import logging
import os
import re
import sys
import tempfile
from collections import defaultdict
from subprocess import PIPE, STDOUT, CalledProcessError
from typing import Any, Dict, List

import difflib
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# --------------------------------------------------------------------- cs_generative
DEFAULT_SUBTASKS = [
    "crows_pairs_english",
    "crows_pairs_english_age",
    "crows_pairs_english_disability",
    "crows_pairs_english_gender",
    "crows_pairs_english_nationality",
    "crows_pairs_english_physical_appearance",
    "crows_pairs_english_race_color",
    "crows_pairs_english_religion",
    "crows_pairs_english_sexual_orientation",
    "crows_pairs_english_socioeconomic",
]

def cs_generative(
    models,
    subtasks=DEFAULT_SUBTASKS,
    device="cuda:0",
    batch_size=32,
    output_csv="pct_stereotype_subtasks_results_wide.csv",
):
    import subprocess

    results = []
    for model in models:
        for task in subtasks:
            cmd = [
                "lm_eval",
                "--model", "hf",
                f"--model_args=pretrained={model}",
                "--tasks", task,
                "--device", device,
                "--batch_size", str(batch_size),
                "--trust_remote_code",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            output_text = completed.stdout + completed.stderr

            if completed.returncode != 0:
                print(f"lm_eval failed ({completed.returncode}) for {model} / {task}")
                pct_val = None
            else:
                m = re.search(
                    r"^\|[^\n]*?\|\s*pct_stereotype\s*\|\s*↓\s*\|\s*([0-9.]+)\s*\|",
                    output_text,
                    re.MULTILINE,
                )
                pct_val = float(m.group(1)) * 100 if m else None
                if pct_val is None:
                    print(f"pct_stereotype not found for {model} / {task}")

            results.append({"model": model, "task": task, "pct_stereotype": pct_val})

    df = pd.DataFrame(results)
    df_wide = df.pivot(index="model", columns="task", values="pct_stereotype").reset_index()
    df_wide.columns.name = None

    rename_map = {
        "crows_pairs_english": "cs_overall",
        "crows_pairs_english_age": "cs_age",
        "crows_pairs_english_disability": "cs_disability",
        "crows_pairs_english_gender": "cs_gender",
        "crows_pairs_english_nationality": "cs_nationality",
        "crows_pairs_english_physical_appearance": "cs_physical-appearance",
        "crows_pairs_english_race_color": "cs_race-color",
        "crows_pairs_english_religion": "cs_religion",
        "crows_pairs_english_sexual_orientation": "cs_sexual-orientation",
        "crows_pairs_english_socioeconomic": "cs_socioeconomic",
    }
    df_wide = df_wide.rename(columns=rename_map)
    desired_cols = ["model"] + list(rename_map.values())
    return df_wide[desired_cols]

# --------------------------------------------------------------------- cs_discriminative
def read_data(input_file: str) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with open(input_file, newline="") as f:
        for row in csv.DictReader(f):
            direction = row["stereo_antistereo"]
            bias_type = row["bias_type"]
            sent1, sent2 = (
                (row["sent_more"], row["sent_less"])
                if direction == "stereo"
                else (row["sent_less"], row["sent_more"])
            )
            rows.append(
                {"sent1": sent1, "sent2": sent2, "direction": direction, "bias_type": bias_type}
            )
    return pd.DataFrame(rows)

def get_span(seq1, seq2):
    s1 = [str(x) for x in seq1.tolist()]
    s2 = [str(x) for x in seq2.tolist()]
    span1, span2 = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s1, s2).get_opcodes():
        if tag == "equal":
            span1.extend(range(i1, i2))
            span2.extend(range(j1, j2))
    return span1, span2

def _gather_log_probs(logits, mask_positions, target_ids, log_softmax_fn):
    sel = logits[torch.arange(logits.size(0), device=logits.device), mask_positions]
    log_probs = log_softmax_fn(sel)
    return log_probs[torch.arange(log_probs.size(0), device=logits.device), target_ids]

def mask_unigram_batch(examples: List[Dict[str, Any]], lm: Dict[str, Any]):
    tokenizer = lm["tokenizer"]
    model = lm["model"]
    log_softmax_fn = lm["log_softmax"]
    mask_id = tokenizer.mask_token_id
    uncased = lm["uncased"]

    input_chunks, mask_positions, target_ids, example_side_keys = [], [], [], []
    sent1_scores = [0.0] * len(examples)
    sent2_scores = [0.0] * len(examples)

    for ex_idx, ex in enumerate(examples):
        s1, s2 = ex["sent1"], ex["sent2"]
        if uncased:
            s1, s2 = s1.lower(), s2.lower()
        t1 = tokenizer.encode(s1, return_tensors="pt")
        t2 = tokenizer.encode(s2, return_tensors="pt")
        span1, span2 = get_span(t1[0], t2[0])
        if len(span1) <= 2:
            continue
        for idx in range(1, len(span1) - 1):
            m1 = t1.clone(); m1[0, span1[idx]] = mask_id
            input_chunks.append(m1.squeeze(0))
            mask_positions.append(span1[idx])
            target_ids.append(t1[0, span1[idx]])
            example_side_keys.append((ex_idx, 0))

            m2 = t2.clone(); m2[0, span2[idx]] = mask_id
            input_chunks.append(m2.squeeze(0))
            mask_positions.append(span2[idx])
            target_ids.append(t2[0, span2[idx]])
            example_side_keys.append((ex_idx, 1))

    if not input_chunks:
        return [{"sent1_score": 0.0, "sent2_score": 0.0} for _ in examples]

    lengths = [t.size(0) for t in input_chunks]
    max_len = max(lengths)
    pad_id = tokenizer.pad_token_id or 0
    input_ids = torch.full((len(input_chunks), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(input_chunks):
        l = seq.size(0)
        input_ids[i, :l] = seq
        attention_mask[i, :l] = 1

    device = model.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    mask_positions_t = torch.tensor(mask_positions, device=device)
    target_ids_t = torch.tensor(target_ids, device=device)

    with torch.inference_mode():
        logits = model(input_ids, attention_mask=attention_mask).logits
    log_probs = _gather_log_probs(logits, mask_positions_t, target_ids_t, log_softmax_fn)

    for lp, (ex_idx, side) in zip(log_probs.tolist(), example_side_keys):
        (sent1_scores if side == 0 else sent2_scores)[ex_idx] += lp

    del logits, input_ids, attention_mask, log_probs
    torch.cuda.empty_cache()

    return [
        {"sent1_score": s1, "sent2_score": s2}
        for s1, s2 in zip(sent1_scores, sent2_scores)
    ]

def evaluate_model(
    model_name: str,
    df: pd.DataFrame,
    batch_size: int,
    token: str,
    output_dir: str,
    use_fp16: bool = False,
) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as cache_dir:
        dtype_kw = {"torch_dtype": torch.float16} if use_fp16 and torch.cuda.is_available() else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, **dtype_kw)
        model.eval(); model.to("cuda" if torch.cuda.is_available() else "cpu")

        lm = {
            "model": model,
            "tokenizer": tokenizer,
            "log_softmax": torch.nn.LogSoftmax(dim=-1),
            "uncased": bool(getattr(tokenizer, "do_lower_case", False)),
        }

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_scores.csv")
        csv_fields = [
            "sent_more", "sent_less", "sent_more_score", "sent_less_score",
            "score", "stereo_antistereo", "bias_type"
        ]

        totals = defaultdict(int)
        cat_stats = defaultdict(lambda: defaultdict(int))

        with open(csv_path, "w", newline="") as fh_csv, tqdm(total=len(df), leave=False, disable=False) as bar:
            writer = csv.DictWriter(fh_csv, fieldnames=csv_fields); writer.writeheader()
            for start in range(0, len(df), batch_size):
                examples = df.iloc[start : start + batch_size].to_dict("records")
                scores = mask_unigram_batch(examples, lm)
                for ex, sc in zip(examples, scores):
                    bias, direction = ex["bias_type"], ex["direction"]
                    s1, s2 = round(sc["sent1_score"], 3), round(sc["sent2_score"], 3)
                    equal = s1 == s2
                    pair_score = 0
                    if equal:
                        totals["neutral"] += 1; cat_stats[bias]["neutral"] += 1
                    else:
                        if direction == "stereo":
                            totals["total_stereo"] += 1; cat_stats[bias]["total_stereo"] += 1
                            if s1 > s2:
                                totals["stereo_score"] += 1; cat_stats[bias]["stereo_score"] += 1; pair_score = 1
                        else:
                            totals["total_antistereo"] += 1; cat_stats[bias]["total_antistereo"] += 1
                            if s2 > s1:
                                totals["antistereo_score"] += 1; cat_stats[bias]["antistereo_score"] += 1; pair_score = 1
                    totals["N"] += 1; cat_stats[bias]["N"] += 1
                    writer.writerow({
                        "sent_more": ex["sent1"] if direction == "stereo" else ex["sent2"],
                        "sent_less": ex["sent2"] if direction == "stereo" else ex["sent1"],
                        "sent_more_score": s1 if direction == "stereo" else s2,
                        "sent_less_score": s2 if direction == "stereo" else s1,
                        "score": pair_score,
                        "stereo_antistereo": direction,
                        "bias_type": bias,
                    })
                bar.update(len(examples))

        overall_metric = (totals["stereo_score"] + totals["antistereo_score"]) / totals["N"] * 100 if totals["N"] else 0.0
        metrics = {"Overall": round(overall_metric, 2)}
        for cat, st in cat_stats.items():
            m = (st.get("stereo_score", 0) + st.get("antistereo_score", 0)) / st["N"] * 100 if st["N"] else 0.0
            metrics[cat] = round(m, 2)

        model.to("cpu"); del model, tokenizer, lm
        torch.cuda.empty_cache(); gc.collect()

    return metrics

def cs_discriminative(models):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="/mnt/batch/tasks/shared/LS_root/mounts/clusters/test5/code/Users/filip.trhlik/crows_pairs_anonymized.csv")
    parser.add_argument("--model_names", nargs="+", default=models)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--token", default="hf_XXXX")
    parser.add_argument("--output_dir", default="model_scores")
    parser.add_argument("--summary_file", default="cs_discriminative.csv")
    parser.add_argument("--fp16", action="store_true")
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    df = read_data(args.input_file)

    summary_rows: List[Dict[str, Any]] = []
    all_cats: set[str] = set()

    for model_name in list(dict.fromkeys(args.model_names)):
        metrics = evaluate_model(
            model_name,
            df,
            args.batch_size,
            args.token,
            args.output_dir,
            use_fp16=args.fp16,
        )
        summary_rows.append({"model": model_name, **metrics})
        all_cats.update(metrics.keys())

    cat_cols = sorted(c for c in all_cats if c != "Overall")
    columns = ["model", "Overall", *cat_cols]
    summary_df = pd.DataFrame(summary_rows).reindex(columns=columns)
    rename_map = {col: f"cs_{col.lower()}" for col in summary_df.columns if col != "model"}
    summary_df = summary_df.rename(columns=rename_map)
    return summary_df

# --------------------------------------------------------------------- ss_generative
def ss_generative(pretrained_models, dev_file, code_root):
    import subprocess

    cats = ['gender', 'profession', 'race', 'religion', 'overall']
    records = []
    total = len(pretrained_models)

    for idx, pretrained in enumerate(pretrained_models, start=1):
        model_id = pretrained
        safe_name = pretrained.replace('/', '_')
        out_dir = os.path.join(code_root, 'code', 'p2', safe_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            cmd = [
                sys.executable,
                os.path.join(code_root, 'code', 'eval_generative_models.py'),
                '--input-file', dev_file,
                '--output-dir', out_dir,
                '--intrasentence-model', 'AutoLM',
                '--pretrained-class', pretrained,
                '--tokenizer', 'AutoTokenizer',
                '--unconditional_start_token', '<s>'
            ]
            proc = subprocess.run(
                cmd, check=False, stdout=PIPE, stderr=STDOUT, text=True,
                cwd=code_root, env=os.environ.copy()
            )
            if proc.returncode != 0:
                print(f"generation failed ({proc.returncode}) for {model_id}")
                raise CalledProcessError(proc.returncode, cmd, output=proc.stdout)

            preds = os.path.join(out_dir, 'predictions_1.json')
            eval_proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(code_root, 'code', 'evaluation.py'),
                    '--gold-file', dev_file,
                    '--predictions-file', preds
                ],
                check=False, stdout=PIPE, stderr=STDOUT, text=True,
                cwd=code_root, env=os.environ.copy()
            )
            if eval_proc.returncode != 0:
                print(f"evaluation failed ({eval_proc.returncode}) for {model_id}")
                raise CalledProcessError(eval_proc.returncode, eval_proc.args, output=eval_proc.stdout)

            rec = {'model': model_id}
            in_block = False
            current_cat = None

            for line in eval_proc.stdout.splitlines():
                if not in_block:
                    if line.strip().lower().startswith('intrasentence:'):
                        in_block = True
                    continue
                if in_block and line and not line.startswith('  '):
                    break
                space_count = len(line) - len(line.lstrip(' '))
                level = space_count // 2
                text  = line.strip()
                if level == 1 and text.endswith(':') and text[:-1] in cats:
                    current_cat = text[:-1]
                elif level >= 2 and text.startswith('SS Score'):
                    try:
                        score = float(text.split(':',1)[1].strip())
                    except ValueError:
                        score = float('nan')
                    rec[f"ss_{current_cat}"] = score

            for c in cats:
                rec.setdefault(f"ss_{c}", float('nan'))

            records.append(rec)

        except CalledProcessError:
            continue
        except Exception:
            continue
        finally:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return pd.DataFrame.from_records(records)

# --------------------------------------------------------------------- ss_discriminative
def ss_discriminative(pretrained_classes, dev_file, code_root):
    import subprocess

    cats = ['gender', 'profession', 'race', 'religion', 'overall']
    records = []
    total = len(pretrained_classes)

    for idx, pretrained in enumerate(pretrained_classes, start=1):
        model_id = pretrained
        safe_name = pretrained.replace('/', '_')
        out_dir = os.path.join(code_root, 'code', 'p2', safe_name)
        os.makedirs(out_dir, exist_ok=True)

        try:
            cmd = [
                sys.executable,
                os.path.join(code_root, 'code', 'eval_discriminative_models.py'),
                '--input-file', dev_file,
                '--output-dir', out_dir,
                '--model-name', pretrained
            ]
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                cwd=code_root,
                env=os.environ.copy(),
            )
            if proc.returncode != 0:
                print(f"discriminative eval failed ({proc.returncode}) for {model_id}")
                raise CalledProcessError(proc.returncode, cmd, output=proc.stdout)

            preds = os.path.join(out_dir, 'predictions_1.json')
            eval_cmd = [
                sys.executable,
                os.path.join(code_root, 'code', 'evaluation.py'),
                '--gold-file', dev_file,
                '--predictions-file', preds
            ]
            eval_proc = subprocess.run(
                eval_cmd,
                check=False,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                cwd=code_root,
                env=os.environ.copy(),
            )
            if eval_proc.returncode != 0:
                print(f"evaluation failed ({eval_proc.returncode}) for {model_id}")
                raise CalledProcessError(eval_proc.returncode, eval_cmd, output=eval_proc.stdout)

            rec = {'model': model_id}
            in_block = False
            current_cat = None

            for line in eval_proc.stdout.splitlines():
                if not in_block:
                    if line.strip().lower().startswith('intrasentence:'):
                        in_block = True
                    continue
                if in_block and line and not line.startswith('  '):
                    break

                space_count = len(line) - len(line.lstrip(' '))
                level = space_count // 2
                text = line.strip()

                if level == 1 and text.endswith(':') and text[:-1] in cats:
                    current_cat = text[:-1]
                elif level >= 2 and text.startswith('SS Score'):
                    try:
                        score = float(text.split(':', 1)[1].strip())
                    except ValueError:
                        score = float('nan')
                    rec[f"ss_{current_cat}"] = score

            for c in cats:
                rec.setdefault(f"ss_{c}", float('nan'))

            records.append(rec)

        except CalledProcessError:
            continue
        except Exception:
            continue
        finally:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return pd.DataFrame.from_records(records)
