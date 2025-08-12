# original code by Marion Bartl with extra fixes for modern compatibility

import argparse
import math
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from nltk import sent_tokenize
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
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

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, help="language: EN or DE")
    p.add_argument("--eval", required=True, help="BEC-Pro / EEC .tsv for evaluation")
    p.add_argument("--tune", help="GAP-flipped .tsv for fine-tuning")
    p.add_argument("--out", required=True, help="output filename stem")
    p.add_argument("--model", help="HF model name (default per language)")
    p.add_argument("--batch", type=int, default=16, help="batch size for FT")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", default="cased_ori_lr_2e-5_collect")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    return p.parse_args()

def save_model(model, epoch, tokenizer, args):
    out_dir = f"./model_save/{args.save_path}/epoch_{epoch}/"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving model to {out_dir}")
    (model.module if hasattr(model, "module") else model).save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(
        [epoch, args.learning_rate, args.adam_epsilon],
        os.path.join(out_dir, "training_args.bin"),
    )

def fine_tune(model, train_dl, epochs, tokenizer, device, args):
    model.to(device).train()
    opt = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=len(train_dl) * epochs
    )
    for ep in range(epochs):
        print(f"\n======== Epoch {ep + 1} / {epochs} ========\nTraining...")
        t0, tot_loss = time.time(), 0.0
        for step, batch in enumerate(train_dl):
            if step and step % 40 == 0:
                print(f"  Batch {step:>5}/{len(train_dl):<5}  Elapsed: {format_time(time.time()-t0)}")
            ids, attn = batch
            ids, labels = mask_tokens(ids, tokenizer)
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            model.zero_grad()
            loss = model(ids, attention_mask=attn, labels=labels).loss
            tot_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        save_model(model, ep + 1, tokenizer, args)
        print(f"\n  Average loss: {tot_loss / len(train_dl):.2f}\n  Epoch took : {format_time(time.time() - t0)}")
    print("Fine-tuning complete!")
    return model

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(args.seed)
    else:
        print("Using CPU.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    pretrained = args.model or ("bert-base-uncased" if args.lang == "EN" else "bert-base-german-dbmdz-cased")
    print("-- Prepare evaluation data --")
    eval_df = pd.read_csv(args.eval, sep="\t")
    print("-- Import BERT model --")
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    for attr in ["_pad_token", "_mask_token", "_cls_token", "_sep_token"]:
        public = attr[1:]
        if not hasattr(tokenizer, attr) and hasattr(tokenizer, public):
            setattr(tokenizer, attr, getattr(tokenizer, public))
    model = BertForMaskedLM.from_pretrained(
        pretrained, output_attentions=False, output_hidden_states=False
    )
    print("-- Calculate associations before fine-tuning --")
    t0 = time.time()
    pre_assoc = model_evaluation(eval_df, tokenizer, model, device)
    print(f"Took {(time.time()-t0)/60:.2f} min")
    eval_df = eval_df.assign(Pre_Assoc=pre_assoc)
    if args.tune:
        if "gap" not in args.tune.lower():
            raise ValueError("Only GAP corpus is supported for fine-tuning.")
        print("-- Import fine-tuning data --")
        gap = pd.read_csv(args.tune, sep="\t")
        corpus = [s for text in gap.Text for s in sent_tokenize(text)]
        max_len = int(2 ** math.ceil(math.log2(max(len(s.split()) for s in corpus))))
        print(f"Max len tuning: {max_len}")
        tune_ids, tune_attn = input_pipeline(corpus, tokenizer, max_len)
        from lib.collection import CollectArguments, collect_embeddings
        STATS_DIR = "bert_GB_2_collect"
        base_args = dict(
            do_collect=True,
            batch_size=16,
            device=device.type,
            model_ckpt_idx=None,
            save_every=1024,
            single=False,
            stats_dir=STATS_DIR,
            verbose=False,
        )
        model_means = BertForMaskedLM.from_pretrained(
            pretrained, output_attentions=False, output_hidden_states=False
        ).to(device)
        collect_embeddings(CollectArguments(**base_args, stage="means"), model_means, tune_ids)
        model_decs = BertForMaskedLM.from_pretrained(
            pretrained, output_attentions=False, output_hidden_states=False
        ).to(device)
        collect_embeddings(CollectArguments(**base_args, stage="decs"), model_decs, tune_ids)
        torch.set_grad_enabled(True)
        model = BertForMaskedLM.from_pretrained(
            pretrained, output_attentions=False, output_hidden_states=False
        ).to(device)
        train_dl = DataLoader(
            TensorDataset(tune_ids, tune_attn),
            sampler=RandomSampler(tune_ids),
            batch_size=args.batch,
        )
        print("-- Set up model fine-tuning --")
        model = fine_tune(model, train_dl, 3, tokenizer, device, args)
        print("-- Calculate associations after fine-tuning --")
        post_assoc = model_evaluation(eval_df, tokenizer, model, device)
        eval_df = eval_df.assign(Post_Assoc=post_assoc)
    else:
        print("No fine-tuning requested.")
    out_path = f"{args.out}_{args.lang}.csv"
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    eval_df.to_csv(out_path, sep="\t", encoding="utf-8", index=False)
    print(f"Results saved to {out_path}")
    if "Prof_Gender" in eval_df.columns:
        m_df, f_df = (
            eval_df[eval_df.Prof_Gender == "male"],
            eval_df[eval_df.Prof_Gender == "female"],
        )
        print("-- Statistics Before --")
        statistics(f_df.Pre_Assoc, m_df.Pre_Assoc)
        if args.tune:
            print("-- Statistics After --")
            statistics(f_df.Post_Assoc, m_df.Post_Assoc)
    else:
        print("No Prof_Gender column – skipping statistics.")
