import sys, types, os, tempfile, importlib.machinery, argparse
from pathlib import Path
import shutil
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as hf_log
from huggingface_hub import HfApi, create_repo

sk_stub = types.ModuleType("sklearn")
sk_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
sk_metrics = types.ModuleType("sklearn.metrics")
sk_metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
sk_metrics.roc_curve = lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("sklearn stub"))
sk_stub.metrics = sk_metrics
sys.modules.update({"sklearn": sk_stub, "sklearn.metrics": sk_metrics})

try:
    from safetensors.torch import safe_open
except ImportError:
    safe_open = None
import torch
import torch.nn as nn
hf_log.set_verbosity_error()

def _widest_rows(pt_obj, hidden):
    best = 0
    def scan(d):
        nonlocal best
        for v in d.values():
            if getattr(v, "ndim", 0) == 2 and v.shape[1] == hidden:
                best = max(best, v.shape[0])
    if isinstance(pt_obj, dict):
        scan(pt_obj)
        for k in ("state_dict", "model_state_dict", "model"):
            if k in pt_obj and isinstance(pt_obj[k], dict):
                scan(pt_obj[k])
    return best or None

def _detect_vocab_rows(folder: Path, hsize: int) -> int:
    for f in sorted(folder.glob("*.safetensors")) + sorted(folder.glob("*.bin")):
        rows = 0
        if f.suffix == ".safetensors" and safe_open:
            with safe_open(f, framework="pt", device="cpu") as sf:
                rows = max(
                    (sf.get_tensor(k).shape[0]
                     for k in sf.keys()
                     if sf.get_tensor(k).ndim == 2 and sf.get_tensor(k).shape[1] == hsize),
                    default=0,
                )
        elif f.suffix == ".bin":
            rows = _widest_rows(torch.load(f, map_location="cpu", weights_only=False), hsize)
        if rows:
            return rows
    raise RuntimeError("Could not find embedding matrix")

def _scrub(tok, final_vocab):
    for token in ["[BOS]", "[EOS]"]:
        if token in tok.added_tokens_encoder:
            idx = tok.added_tokens_encoder[token]
            del tok.added_tokens_encoder[token]
            if idx in tok.added_tokens_decoder:
                del tok.added_tokens_decoder[idx]
    for attr in ["_added_tokens_encoder", "_added_tokens_decoder"]:
        if hasattr(tok, attr):
            mapping = getattr(tok, attr)
            if isinstance(mapping, dict):
                for token in ["[BOS]", "[EOS]"]:
                    if token in mapping:
                        del mapping[token]
    for attr in ["bos_token", "eos_token"]:
        if hasattr(tok, attr) and getattr(tok, attr) in ["[BOS]", "[EOS]"]:
            setattr(tok, attr, None)
    if hasattr(tok, "init_kwargs"):
        if "added_tokens" in tok.init_kwargs:
            tok.init_kwargs["added_tokens"] = [
                t for t in tok.init_kwargs["added_tokens"]
                if t.get("content") not in ["[BOS]", "[EOS]"]
            ]
        tok.init_kwargs["vocab_size"] = final_vocab
    if hasattr(tok, "special_tokens_map"):
        new_map = {k: v for k, v in tok.special_tokens_map.items() if v not in ["[BOS]", "[EOS]"]}
        if hasattr(tok, "_set_attr"):
            tok._set_attr("special_tokens_map", new_map)
        else:
            tok.special_tokens_map = new_map

def repair_and_publish(ckpt_path, repo_id, hf_token, final_vocab=16384, base_repo="ltg/ltg-bert-babylm"):
    ckpt = Path(ckpt_path).expanduser().resolve()
    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    if cfg.model_type != "ltgbert":
        raise ValueError(f"Unsupported model type: {cfg.model_type}")
    rows = _detect_vocab_rows(ckpt, cfg.hidden_size)
    cfg.vocab_size = rows
    model = AutoModelForMaskedLM.from_pretrained(
        ckpt,
        config=cfg,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if rows > final_vocab:
        model.resize_token_embeddings(final_vocab)
        model.config.vocab_size = final_vocab
        if hasattr(model, "classifier") and hasattr(model.classifier, "nonlinearity"):
            if isinstance(model.classifier.nonlinearity, nn.Sequential) and len(model.classifier.nonlinearity) > 5:
                layer5 = model.classifier.nonlinearity[5]
                if isinstance(layer5, nn.Linear):
                    new_layer = nn.Linear(
                        in_features=layer5.in_features,
                        out_features=final_vocab,
                        bias=layer5.bias is not None
                    )
                    with torch.no_grad():
                        new_layer.weight.copy_(layer5.weight[:final_vocab])
                        if layer5.bias is not None:
                            new_layer.bias.copy_(layer5.bias[:final_vocab])
                    model.classifier.nonlinearity[5] = new_layer
        _scrub(tok, final_vocab)
    tmp = Path(tempfile.mkdtemp(prefix="ltgbert-fixed-"))
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    py_files = list(ckpt.glob("*.py"))
    if not py_files:
        from huggingface_hub import snapshot_download
        local_repo = Path(snapshot_download(base_repo, allow_patterns=["*ltgbert.py"]))
        py_files = list(local_repo.glob("*ltgbert.py"))
    if not py_files:
        raise RuntimeError("Missing ltgbert implementation files")
    for f in py_files:
        shutil.copy(f, tmp)
    (tmp / "__init__.py").write_text(
        "from .configuration_ltgbert import LtgBertConfig\n"
        "from .modeling_ltgbert import LtgBertModel, LtgBertForMaskedLM\n"
    )
    create_repo(repo_id, token=hf_token, exist_ok=True)
    api = HfApi(token=hf_token)
    api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="model", token=hf_token)
    shutil.rmtree(tmp)

def main():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--final_vocab", type=int, default=16384)
    p.add_argument("--base_repo", default="ltg/ltg-bert-babylm")
    a = p.parse_args()
    if not a.hf_token:
        raise SystemExit("HF token not provided (use --hf_token or set HF_TOKEN)")
    repair_and_publish(a.ckpt_path, a.repo_id, a.hf_token, a.final_vocab, a.base_repo)

if __name__ == "__main__":
    main()
