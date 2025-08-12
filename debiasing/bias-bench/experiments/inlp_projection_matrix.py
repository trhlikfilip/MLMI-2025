import argparse
import functools
import inspect
import os
import warnings
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoModelForMaskedLM
from transformers.utils import ModelOutput

from bias_bench.dataset import load_inlp_data
from bias_bench.debias.inlp import compute_projection_matrix
from bias_bench.model import models
from bias_bench.util import generate_experiment_id

# Fallback order for loading checkpoints
MODEL_FALLBACK_ORDER = (AutoModel, AutoModelForMaskedLM)

def load_any_model(checkpoint: str, trust_remote_code: bool):
    last_err = None
    for auto_cls in MODEL_FALLBACK_ORDER:
        try:
            return auto_cls.from_pretrained(
                checkpoint, trust_remote_code=trust_remote_code
            )
        except (ValueError, RuntimeError) as err:
            last_err = err
    raise last_err

POSSIBLE_EXTRA_KWARGS = {
    "token_type_ids",
    "position_ids",
    "head_mask",
    "inputs_embeds",
    "past_key_values",
}

def patch_model_forward(model):
    """Ensure output_hidden_states=True and return a dict-like output."""
    orig_forward = model.forward
    sig = inspect.signature(orig_forward)
    to_strip = POSSIBLE_EXTRA_KWARGS - set(sig.parameters)

    @functools.wraps(orig_forward)
    def wrapped_forward(*args, **kwargs):
        for k in list(kwargs.keys()):
            if k in to_strip:
                kwargs.pop(k)
        kwargs["output_hidden_states"] = True
        outputs = orig_forward(*args, **kwargs)

        if isinstance(outputs, (dict, ModelOutput)):
            out: Dict[str, Any] = (
                dict(outputs.items()) if isinstance(outputs, dict) else outputs.to_dict()
            )
        else:
            out = {"hidden_states": outputs[-1]}

        if "last_hidden_state" not in out:
            hidden = out.get("hidden_states")
            if hidden is None:
                raise RuntimeError("Model did not return hidden_states.")
            out["last_hidden_state"] = hidden[-1]
        return out

    model.forward = wrapped_forward  # type: ignore[attr-defined]
    return model

def main() -> None:
    thisdir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute an INLP projection matrix.",
    )

    # Paths
    parser.add_argument(
        "--persistent_dir",
        default=os.path.realpath(os.path.join(thisdir, "..")),
        help="Where Bias-Bench keeps datasets/results.",
    )

    # Model
    parser.add_argument("--model", default="auto", choices=["auto", "BertModel",
                        "AlbertModel", "RobertaModel", "GPT2Model"])
    parser.add_argument("--model_name_or_path", default="bert-base-uncased")
    parser.add_argument("--trust_remote_code", action="store_true")

    # INLP options
    parser.add_argument("--bias_type", default="gender",
                        choices=["gender", "race", "religion"])
    parser.add_argument("--n_classifiers", type=int, default=80)

    # Data sub-sampling
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="0 < fraction ≤ 1")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # Run ID / save tag
    experiment_id = generate_experiment_id(
        name="projection",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    save_tag = experiment_id.replace(os.sep, "_").replace("/", "_")

    print("Computing projection matrix with settings:")
    for k, v in vars(args).items():
        print(f"  • {k}: {v}")

    # Load debiasing data
    data = load_inlp_data(args.persistent_dir, args.bias_type, seed=args.seed)

    # Optional stratified sub-sampling
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train_fraction must satisfy 0 < f ≤ 1")

    if args.train_fraction < 1.0:
        rng = np.random.default_rng(args.seed)

        # Case A: explicit labels in each split
        def is_explicit_split(split) -> bool:
            if isinstance(split, dict):
                return "labels" in split or "label" in split
            if hasattr(split, "column_names"):
                return "labels" in split.column_names or "label" in split.column_names
            return False

        if any(is_explicit_split(s) for s in data.values()):
            for split_name, split in data.items():
                if isinstance(split, dict):
                    lbl = np.asarray(split.get("labels", split.get("label")))
                elif hasattr(split, "column_names"):
                    col = "labels" if "labels" in split.column_names else "label"
                    lbl = np.asarray(split[col])
                else:
                    lbl = np.asarray([ex.get("labels", ex.get("label")) for ex in split])

                keep: List[int] = []
                for y in np.unique(lbl):
                    idx = np.where(lbl == y)[0]
                    n = max(1, int(round(len(idx) * args.train_fraction)))
                    keep.extend(rng.choice(idx, n, replace=False))
                keep = sorted(keep)

                if isinstance(split, dict):
                    data[split_name] = {k: v[keep] for k, v in split.items()}
                elif hasattr(split, "select"):
                    data[split_name] = split.select(keep)
                else:
                    data[split_name] = [split[i] for i in keep]
                print(f"→ kept {len(keep):5d} / {len(lbl):5d} in '{split_name}'")
        # Case B: simple buckets
        else:
            for bucket, sentences in data.items():
                if not isinstance(sentences, (list, tuple, np.ndarray)):
                    continue
                n_keep = max(1, int(round(len(sentences) * args.train_fraction)))
                idx = rng.choice(len(sentences), n_keep, replace=False)
                idx.sort()
                data[bucket] = [sentences[i] for i in idx]
                print(f"→ kept {len(idx):5d} / {len(sentences):5d} in '{bucket}'")

    # Model & tokenizer
    if args.model.lower() == "auto":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Some weights of the model checkpoint*")
            model = load_any_model(args.model_name_or_path, args.trust_remote_code)
    else:
        model_cls = getattr(models, args.model)
        model = model_cls(args.model_name_or_path)
    model = patch_model_forward(model).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 512

    # Train INLP and save
    P = compute_projection_matrix(
        model,
        tokenizer,
        data,
        bias_type=args.bias_type,
        n_classifiers=args.n_classifiers,
    )

    out_dir = os.path.join(args.persistent_dir, "results", "projection_matrix")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{save_tag}.pt")
    torch.save(P, out_path)
    print("✓ Projection matrix written to:", out_path)

if __name__ == "__main__":
    main()
