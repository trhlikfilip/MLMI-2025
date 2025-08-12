import json
import os
import shutil
import argparse
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from colorama import Back, Fore, Style, init
    init(autoreset=True)
except ImportError:
    class _Dummy:
        def __getattr__(self, name):
            return ""
    Back = Fore = Style = _Dummy()
    def init(*args, **kwargs):
        pass

hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
hub_cache = os.path.join(hf_home, "hub")
if os.path.isdir(hub_cache):
    try:
        shutil.rmtree(hub_cache)
    except Exception as e:
        print(f"Warning: failed to clear HF cache: {e}")

from tqdm import tqdm
import dataloader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

init()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--input-file", default="data/dev.json", type=str)
    parser.add_argument("--output-dir", default="predictions/", type=str)
    parser.add_argument("--output-file", default=None, type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    return parser.parse_args()

class BiasEvaluator:
    def __init__(self, args):
        self.input_file = args.input_file
        self.model_name = args.model_name
        self.device = "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="right", trust_remote_code=True
        )
        self.MASK_TOKEN = self.tokenizer.mask_token
        self.MASK_TOKEN_IDX = self.tokenizer.convert_tokens_to_ids(self.MASK_TOKEN)
        self.batch_size = args.batch_size
        self.max_seq_length = None if self.batch_size == 1 else args.max_seq_length

    def evaluate_intrasentence(self):
        use_masked = True
        try:
            lm = AutoModelForMaskedLM.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(self.device)
        except Exception:
            lm = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(self.device)
            use_masked = False
        lm.eval()

        pad_to_max = self.batch_size > 1
        dataset = dataloader.IntrasentenceLoader(
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            pad_to_max_length=pad_to_max,
            input_file=self.input_file,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size)
        word_probs = defaultdict(list)

        for sent_id, next_token, input_ids, attention_mask, token_type_ids in tqdm(loader):
            if use_masked:
                ids = torch.stack(input_ids).to(self.device).transpose(0, 1)
                masks = torch.stack(attention_mask).to(self.device).transpose(0, 1)
                types = torch.stack(token_type_ids).to(self.device).transpose(0, 1)
                ntok = next_token.to(self.device)
                mask_locs = ids == self.MASK_TOKEN_IDX
                supports_seg = (
                    hasattr(lm, "config")
                    and getattr(lm.config, "type_vocab_size", 0) > 1
                )
                kwargs = {"input_ids": ids, "attention_mask": masks}
                if supports_seg:
                    kwargs["token_type_ids"] = types
                with torch.no_grad():
                    logits = lm(**kwargs)[0]
                    probs = torch.softmax(logits, dim=-1)
                masked_probs = probs[mask_locs]
                sel = masked_probs.index_select(1, ntok.view(-1)).diag()
                for i, p in enumerate(sel):
                    word_probs[sent_id[i]].append(p.item())
            else:
                ids = torch.stack(input_ids).to(self.device)
                ntok = next_token.to(self.device)
                for i in range(ids.size(0)):
                    seq = ids[i]
                    mask_pos = (seq == self.MASK_TOKEN_IDX).nonzero(as_tuple=True)[0]
                    mp = mask_pos.item()
                    prefix = seq[:mp].unsqueeze(0)
                    with torch.no_grad():
                        logits = lm(prefix)[0]
                    token_logits = logits[0, mp - 1]
                    prob = torch.softmax(token_logits, dim=-1)[ntok[i]].item()
                    word_probs[sent_id[i]].append(prob)

        return [{"id": sid, "score": float(np.mean(ps))} for sid, ps in word_probs.items()]

    def evaluate(self):
        return {"intrasentence": self.evaluate_intrasentence()}

if __name__ == "__main__":
    args = parse_args()
    evaluator = BiasEvaluator(args)
    bias_results = evaluator.evaluate()
    fname = args.output_file or "predictions_1.json"
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, fname)
    with open(outpath, "w") as f:
        json.dump(bias_results, f, indent=2)
    print(f"Saved predictions to {outpath}")
