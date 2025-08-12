import argparse, math, os, random, time
import numpy as np, pandas as pd, torch
from nltk import sent_tokenize
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AdamW,
    get_linear_schedule_with_warmup,
)
from bias_utils.utils import (
    model_evaluation,
    mask_tokens,
    input_pipeline,
    format_time,
    statistics,
)

#ADJUSTED FOR LTG-Baseline: 
#Forces trust_remote_code, and is made model-agnostic (for any Hugging Face model)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--tune")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="babylm/ltgbert-100m-2024")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", default="cased_ori_lr_2e-5_collect")
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--adam_epsilon", type=float, default=1e-8)
    return ap.parse_args()

def save_model(model, epoch, tok, args):
    odir = f"./model_save/{args.save_path}/epoch_{epoch}/"
    os.makedirs(odir, exist_ok=True)
    (model.module if hasattr(model, "module") else model).save_pretrained(odir)
    tok.save_pretrained(odir)

def fine_tune(model, dl, epochs, tok, dev, args):
    model.to(dev).train()
    opt = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    sched = get_linear_schedule_with_warmup(opt, 0, len(dl) * epochs)
    for ep in range(epochs):
        print(f"\n======== Epoch {ep+1} / {epochs} ========")
        t0, tot = time.time(), 0.0
        for step, (ids, att) in enumerate(dl):
            if step and step % 40 == 0:
                print(f"  Batch {step:>4}/{len(dl):<4}  Elapsed: {format_time(time.time()-t0)}")
            ids, labs = mask_tokens(ids, tok)
            ids, labs, att = ids.to(dev), labs.to(dev), att.to(dev)
            model.zero_grad()
            loss = model(ids, attention_mask=att, labels=labs).loss
            tot += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        save_model(model, ep+1, tok, args)
        print(f"  Avg loss: {tot/len(dl):.2f}   Epoch time: {format_time(time.time()-t0)}")
    print("Fine-tuning complete!")
    return model

if __name__ == "__main__":
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("-- Prepare evaluation data --")
    eval_df = pd.read_csv(args.eval, sep="\t")
    print("-- Load model & tokenizer --")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, trust_remote_code=True,
        output_attentions=False, output_hidden_states=False
    ).to(dev)
    for a in ["_pad_token", "_mask_token", "_cls_token", "_sep_token"]:
        pub = a[1:]
        if not hasattr(tok, a) and hasattr(tok, pub):
            setattr(tok, a, getattr(tok, pub))
    if tok.mask_token != "[MASK]":
        eval_df = eval_df.applymap(
            lambda x: x.replace("[MASK]", tok.mask_token) if isinstance(x, str) else x
        )
    print("-- Calculate associations before fine-tuning --")
    t0 = time.time()
    pre = model_evaluation(eval_df, tok, model, dev)
    print(f"Took {(time.time()-t0)/60:.2f} min")
    eval_df = eval_df.assign(Pre_Assoc=pre)
    if args.tune:
        if "gap" not in args.tune.lower():
            raise ValueError("Only GAP corpus is supported.")
        print("-- Load GAP corpus --")
        gap = pd.read_csv(args.tune, sep="\t")
        corpus = [s for txt in gap.Text for s in sent_tokenize(txt)]
        max_len = min(
            2 ** math.ceil(math.log2(max(len(s.split()) for s in corpus))),
            tok.model_max_length,
        )
        print(f"Max len tuning: {max_len}")
        ids, att = input_pipeline(corpus, tok, int(max_len))
        ft_model = AutoModelForMaskedLM.from_pretrained(
            args.model, trust_remote_code=True,
            output_attentions=False, output_hidden_states=False
        ).to(dev)
        dl = DataLoader(TensorDataset(ids, att),
                        sampler=RandomSampler(ids),
                        batch_size=args.batch)
        print("-- Fine-tuning --")
        ft_model = fine_tune(ft_model, dl, 3, tok, dev, args)
        print("-- Calculate associations after fine-tuning --")
        post = model_evaluation(eval_df, tok, ft_model, dev)
        eval_df = eval_df.assign(Post_Assoc=post)
    else:
        print("No fine-tuning requested.")
    out_csv = f"{args.out}_{args.lang}.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    eval_df.to_csv(out_csv, sep="\t", encoding="utf-8", index=False)
    print(f"Results saved to {out_csv}")
    if "Prof_Gender" in eval_df.columns:
        male = eval_df[eval_df.Prof_Gender == "male"]
        fem = eval_df[eval_df.Prof_Gender == "female"]
        print("-- Stats before --")
        statistics(fem.Pre_Assoc, male.Pre_Assoc)
        if args.tune:
            print("-- Stats after --")
            statistics(fem.Post_Assoc, male.Post_Assoc)
    else:
        print("No Prof_Gender column – skipping stats.")
