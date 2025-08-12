import argparse, math, os, random, time
from types import MethodType, SimpleNamespace
import numpy as np, pandas as pd, torch
from nltk import sent_tokenize
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup
from transformers.activations import gelu_new as _orig_gelu_new
from bias_utils.utils import model_evaluation, mask_tokens, input_pipeline, format_time, statistics

#ADJUSTED FOR LTG-BERT: 
# LTG-specific safety patches (safe GELU, SafeMaskedSoftmax, bucket and position clamps), ensures compatibility with non-BERT MLM architectures via ensure_bert_like,
# and includes a shrink_and_save step to restore BabyLM’s original vocabulary size before saving

def _safe_gelu(x: torch.Tensor):
    return _orig_gelu_new(torch.clamp(x, -10_000.0, 10_000.0))

import transformers.activations as _act
_act.gelu_new = _safe_gelu

def ensure_bert_like(model):
    if getattr(model, "cls", None) and getattr(model.cls, "predictions", None) and getattr(model.cls.predictions, "decoder", None):
        return model
    decoder = None
    if callable(getattr(model, "get_output_embeddings", None)):
        decoder = model.get_output_embeddings()
    if decoder is None:
        for cand in ["lm_head", "mlm_head", "predictions", "generator_lm_head"]:
            if hasattr(model, cand):
                head = getattr(model, cand)
                decoder = getattr(head, "decoder", head)
                break
    if decoder is None:
        return model
    if isinstance(decoder, torch.nn.Parameter):
        decoder = SimpleNamespace(weight=decoder)
    if not hasattr(decoder, "parameters"):
        decoder.parameters = lambda self=decoder: [getattr(self, "weight", None)]
    w = getattr(decoder, "weight", None)
    if w is not None and w.ndim == 2:
        decoder.out_features, decoder.in_features = w.shape
    model.cls = SimpleNamespace(predictions=SimpleNamespace(decoder=decoder))
    return model

try:
    import lib.model as _lib_model
    _orig_strip = _lib_model.strip_model_bert
    _lib_model.strip_model_bert = lambda m, d: _orig_strip(ensure_bert_like(m), d)
except Exception:
    pass

class SafeMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mask, dim):
        zero_mask = (mask.sum(dim=dim, keepdim=True) == 0)
        probs = torch.softmax(scores.masked_fill(mask == 0, float("-inf")).masked_fill(zero_mask, 0.0), dim)
        ctx.save_for_backward(probs, mask)
        ctx.dim = dim
        return probs
    @staticmethod
    def backward(ctx, grad):
        probs, mask = ctx.saved_tensors
        dim = ctx.dim
        grad = grad * probs - probs * (grad * probs).sum(dim=dim, keepdim=True)
        return grad.masked_fill(mask == 0, 0.0), None, None

def _safe_bucket(self, rel_pos, bucket, max_pos):
    pos = self.__class__.make_log_bucket_position(self, rel_pos, bucket, max_pos)
    return pos.clamp(min=-(bucket - 1), max=bucket - 1)

def _clamp_position_indices(mod, _):
    if hasattr(mod, "position_indices"):
        n = mod.position_indices.size(-1)
        mod.position_indices.clamp_(0, n - 1)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--tune")
    p.add_argument("--out", required=True)
    p.add_argument("--model")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    return p.parse_args()

def fine_tune(model, dl, epochs, tok, device, lr, eps):
    model.to(device).train()
    opt = AdamW(model.parameters(), lr=lr, eps=eps)
    sched = get_linear_schedule_with_warmup(opt, 0, len(dl) * epochs)
    for ep in range(1, epochs + 1):
        print(f"Epoch {ep}/{epochs}")
        t0 = time.time()
        tot = 0.0
        for step, (ids, attn) in enumerate(dl, 1):
            if step % 40 == 0:
                print(f"{step}/{len(dl)} elapsed {format_time(time.time() - t0)}")
            ids, labels = mask_tokens(ids, tok)
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            loss = model(ids, attention_mask=attn, labels=labels).loss
            if not torch.isfinite(loss):
                print("NaN/Inf loss – batch skipped")
                model.zero_grad()
                continue
            loss.backward()
            tot += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model.zero_grad()
        print("Avg loss", tot / len(dl), "time", format_time(time.time() - t0))
    return model

BABYLM_ORIG = 16384

def shrink_and_save(model, tok, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    is_babylm = hasattr(model.config, "model_type") and "ltgbert" in model.config.model_type.lower()
    needs_trim = is_babylm and len(tok) > BABYLM_ORIG
    if needs_trim:
        from tokenizers.models import WordPiece
        wp_old: WordPiece = tok.backend_tokenizer.model
        keep_tokens = [t for t, i in sorted(tok.get_vocab().items(), key=lambda x: x[1])][:BABYLM_ORIG]
        tok.backend_tokenizer.model = WordPiece(
            vocab={t: i for i, t in enumerate(keep_tokens)},
            unk_token=wp_old.unk_token,
            continuing_subword_prefix=wp_old.continuing_subword_prefix,
            max_input_chars_per_word=wp_old.max_input_chars_per_word,
        )
        tok.additional_special_tokens = []
        tok.bos_token = tok.eos_token = None
        model.resize_token_embeddings(BABYLM_ORIG, mean_resizing=False)
        if hasattr(model, "classifier") and hasattr(model.classifier, "nonlinearity"):
            lin = model.classifier.nonlinearity[5]
            lin.weight = torch.nn.Parameter(lin.weight[:BABYLM_ORIG].clone())
            lin.bias = torch.nn.Parameter(lin.bias[:BABYLM_ORIG].clone())
            lin.out_features = BABYLM_ORIG
        try:
            model.tie_weights()
        except Exception:
            pass
        model.config.vocab_size = BABYLM_ORIG
        print(f"Shrunk back to {BABYLM_ORIG} tokens before saving")
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print("Model + tokenizer saved to", save_dir)

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    pretrained = args.model or ("bert-base-uncased" if args.lang.upper() == "EN" else "bert-base-german-dbmdz-cased")
    eval_df = pd.read_csv(args.eval, sep="\t")
    tok = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
    for attr in ["_pad_token", "_mask_token", "_cls_token", "_sep_token"]:
        pub = attr[1:]
        setattr(tok, attr, getattr(tok, pub, getattr(tok, pub, None)))
    model = ensure_bert_like(AutoModelForMaskedLM.from_pretrained(
        pretrained, trust_remote_code=True, output_attentions=False, output_hidden_states=False
    ))
    for m in model.modules():
        if hasattr(m, "make_log_bucket_position"):
            m.make_log_bucket_position = MethodType(_safe_bucket, m)
        if hasattr(m, "MaskedSoftmax"):
            m.MaskedSoftmax = SafeMaskedSoftmax
        if hasattr(m, "position_indices"):
            m.register_forward_pre_hook(_clamp_position_indices)
    if len(tok) > model.config.vocab_size:
        old, new = model.config.vocab_size, len(tok)
        model.resize_token_embeddings(new)
        model.config.vocab_size = new
        print(f"Resized token embeddings {old} -> {new}")
    print("Calculating associations before fine-tuning")
    t0 = time.time()
    eval_df["Pre_Assoc"] = model_evaluation(eval_df, tok, model, device)
    print("Done in", format_time(time.time() - t0))
    if args.tune:
        gap = pd.read_csv(args.tune, sep="\t")
        corpus = [s for txt in gap.Text for s in sent_tokenize(txt)]
        max_len = 2 ** math.ceil(math.log2(max(len(s.split()) for s in corpus)))
        ids, attn = input_pipeline(corpus, tok, max_len)
        dl = DataLoader(TensorDataset(ids, attn), sampler=RandomSampler(ids), batch_size=args.batch)
        model = fine_tune(model, dl, 3, tok, device, args.learning_rate, args.adam_epsilon)
        print("Calculating associations after fine-tuning")
        eval_df["Post_Assoc"] = model_evaluation(eval_df, tok, model, device)
    else:
        print("No fine-tuning requested")
    csv_path = f"{args.out}_{args.lang}.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    eval_df.to_csv(csv_path, sep="\t", index=False)
    print("Results saved to", csv_path)
    save_dir = f"{args.out}_{args.lang}_model"
    shrink_and_save(model, tok, save_dir)
    if "Prof_Gender" in eval_df.columns:
        m_df = eval_df[eval_df.Prof_Gender == "male"]
        f_df = eval_df[eval_df.Prof_Gender == "female"]
        print("Statistics Before")
        statistics(f_df.Pre_Assoc, m_df.Pre_Assoc)
        if args.tune:
            print("Statistics After")
            statistics(f_df.Post_Assoc, m_df.Post_Assoc)
