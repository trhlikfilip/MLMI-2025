#Edited to allow all Hugging Face models
import sys
import os
import shutil
try:
    from colorama import Back, Fore, Style, init
    init(autoreset=True)
except ImportError:
    class _Dummy:
        def __getattr__(self, name): return ""
    Back = Fore = Style = _ummy = _Dummy()
    def init(*args, **kwargs): pass

hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
hub_cache = os.path.join(hf_home, "hub")
if os.path.isdir(hub_cache):
    try:
        shutil.rmtree(hub_cache)
    except Exception as e:
        print(f"Warning: failed to clear HF cache: {e}")

import json
from argparse import ArgumentParser
import numpy as np
import torch
import transformers
import numpy
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataloader
from intersentence_loader import IntersentenceDataset
from models import models

init()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained-class", default="gpt2", type=str)
    parser.add_argument("--no-cuda", default=False, action="store_true")
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--input-file", default="../data/dev.json", type=str)
    parser.add_argument("--output-dir", default="predictions/", type=str)
    parser.add_argument("--cache-dir", default=None, type=str)
    parser.add_argument("--intrasentence-model", default="GPT2LM", type=str)
    parser.add_argument("--intrasentence-load-path", default=None)
    parser.add_argument("--intersentence-model", default="ModelNSP", type=str)
    parser.add_argument("--intersentence-load-path", default=None)
    parser.add_argument("--tokenizer", default="GPT2Tokenizer", type=str)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--unconditional_start_token", default="<|endoftext|>", type=str)
    parser.add_argument("--skip-intersentence", default=False, action="store_true")
    parser.add_argument("--skip-intrasentence", default=False, action="store_true")
    parser.add_argument("--small", default=False, action="store_true")
    return parser.parse_args()

class BiasEvaluator(object):
    def __init__(
        self,
        pretrained_class="gpt2",
        no_cuda=False,
        batch_size=51,
        input_file="data/bias.json",
        intrasentence_model="GPT2LM",
        intrasentence_load_path=None,
        intersentence_model="ModelNSP",
        intersentence_load_path=None,
        tokenizer="GPT2Tokenizer",
        unconditional_start_token="<|endoftext|>",
        skip_intrasentence=False,
        skip_intersentence=False,
        max_seq_length=64,
        small=False,
        output_dir="predictions/",
        cache_dir=None,
    ):
        self.BATCH_SIZE = batch_size
        filename = os.path.abspath(input_file)
        self.dataloader = dataloader.StereoSet(filename)
        self.cuda = not no_cuda
        self.device = "cuda" if self.cuda else "cpu"
        self.SKIP_INTERSENTENCE = skip_intersentence
        self.SKIP_INTRASENTENCE = skip_intrasentence
        self.UNCONDITIONAL_START_TOKEN = unconditional_start_token
        self.CACHE_DIR = cache_dir
        self.hf_kwargs = {}
        if self.CACHE_DIR:
            self.hf_kwargs["cache_dir"] = self.CACHE_DIR
        self.PRETRAINED_CLASS = pretrained_class
        self.TOKENIZER = tokenizer
        try:
            self.tokenizer = getattr(transformers, self.TOKENIZER).from_pretrained(
                self.PRETRAINED_CLASS,
                **self.hf_kwargs
            )
        except (OSError, EnvironmentError):
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.INTRASENTENCE_MODEL = intrasentence_model
        self.INTRASENTENCE_LOAD_PATH = intrasentence_load_path
        self.INTERSENTENCE_MODEL = intersentence_model
        self.INTERSENTENCE_LOAD_PATH = intersentence_load_path
        self.max_seq_length = max_seq_length

    def _get_start_token_ids(self):
        if getattr(self.tokenizer, 'bos_token_id', None) is not None:
            return [self.tokenizer.bos_token_id]
        if getattr(self.tokenizer, 'cls_token_id', None) is not None:
            return [self.tokenizer.cls_token_id]
        ids = self.tokenizer.encode(
            self.UNCONDITIONAL_START_TOKEN,
            add_special_tokens=False
        )
        if len(ids) >= 1:
            return ids
        raise ValueError("Could not determine a valid start token.")

    def evaluate_intrasentence(self):
        model = getattr(models, self.INTRASENTENCE_MODEL)(
            self.PRETRAINED_CLASS,
            **self.hf_kwargs
        ).to(self.device)
        model.eval()
        start_ids = self._get_start_token_ids()
        start_token = torch.tensor(start_ids, device=self.device).unsqueeze(0)
        initial_token_probabilities = model(start_token)
        initial_token_probabilities = torch.softmax(initial_token_probabilities[0], dim=-1)
        clusters = self.dataloader.get_intrasentence_examples()
        predictions = []
        for cluster in tqdm(clusters):
            for sentence in cluster.sentences:
                tokens = self.tokenizer.encode(sentence.sentence)
                joint_sentence_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()
                ]
                tokens_tensor = torch.tensor(tokens, device=self.device).unsqueeze(0)
                output = torch.softmax(model(tokens_tensor)[0], dim=-1)
                for idx in range(1, len(tokens)):
                    joint_sentence_probability.append(
                        output[0, idx-1, tokens[idx]].item()
                    )
                score = np.mean([np.log2(p) for p in joint_sentence_probability])
                score = 2**score
                predictions.append({'id': sentence.ID, 'score': score})
        return predictions

    def evaluate_intersentence(self):
        model = getattr(models, self.INTERSENTENCE_MODEL)(
            self.PRETRAINED_CLASS,
            **self.hf_kwargs
        ).to(self.device)
        start_ids = self._get_start_token_ids()
        start_token = torch.tensor(start_ids, device=self.device).unsqueeze(0)
        initial_token_probabilities = model(start_token)
        initial_token_probabilities = torch.softmax(initial_token_probabilities[0], dim=-1)
        model.eval()
        clusters = self.dataloader.get_intersentence_examples()[:1000]
        predictions = []
        for cluster in tqdm(clusters):
            context = cluster.context
            if context and context[-1] not in ['.', '!', '?']:
                context += '.'
            for sentence in cluster.sentences:
                full = f"{context} {sentence.sentence}"
                tokens = self.tokenizer.encode(full)
                tokens_tensor = torch.tensor(tokens, device=self.device).unsqueeze(0)
                context_len = len(self.tokenizer.encode(context))
                sent_prob = [initial_token_probabilities[0,0,tokens[context_len]].item()]
                ctxt_prob = [initial_token_probabilities[0,0,tokens[0]].item()]
                output = torch.softmax(model(tokens_tensor)[0], dim=-1)
                for i in range(1, context_len):
                    ctxt_prob.append(output[0,i-1,tokens[i]].item())
                for i in range(1, len(tokens)):
                    sent_prob.append(output[0,i-1,tokens[i]].item())
                bare = self.tokenizer.encode(sentence.sentence)
                bare_tensor = torch.tensor(bare, device=self.device).unsqueeze(0)
                no_ctx = [initial_token_probabilities[0,0,bare[0]].item()]
                output2 = torch.softmax(model(bare_tensor)[0], dim=-1)
                for i in range(1, len(bare)):
                    no_ctx.append(output2[0,i-1,bare[i]].item())
                ctx_score = np.mean([np.log2(p) for p in ctxt_prob])
                no_ctx_score = np.mean([np.log2(p) for p in no_ctx])
                overall = no_ctx_score / ctx_score
                predictions.append({'id': sentence.ID, 'score': overall})
        return predictions

    def evaluate_nsp_intersentence(self):
        nsp_dim = 300
        model = getattr(models, self.INTERSENTENCE_MODEL)(
            self.PRETRAINED_CLASS,
            nsp_dim=nsp_dim,
            **self.hf_kwargs
        ).to(self.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if "gpt2" in self.PRETRAINED_CLASS.lower():
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            model.core_model.resize_token_embeddings(len(self.tokenizer))
        model = torch.nn.DataParallel(model)
        if self.INTERSENTENCE_LOAD_PATH:
            model.load_state_dict(torch.load(self.INTERSENTENCE_LOAD_PATH))
        model.eval()
        dataset = IntersentenceDataset(self.tokenizer, args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        preds = []
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids, token_type_ids, attention_mask, sentence_id = batch
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = torch.softmax(outputs, dim=1)
            for i in range(input_ids.size(0)):
                score = outputs[i, 0 if "bert" in self.PRETRAINED_CLASS.lower() else 1].item()
                preds.append({'id': sentence_id[i], 'score': score})
        return preds

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate(self):
        bias = {}
        if not self.SKIP_INTRASENTENCE:
            bias['intrasentence'] = self.evaluate_intrasentence()
        if self.SKIP_INTERSENTENCE:
            if self.INTERSENTENCE_MODEL == "ModelNSP":
                bias['intersentence'] = self.evaluate_nsp_intersentence()
            else:
                bias['intersentence'] = self.evaluate_intersentence()
        return bias

if __name__ == "__main__":
    args = parse_args()
    evaluator = BiasEvaluator(**vars(args))
    results = evaluator.evaluate()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"predictions_1.json")
    with open(output_file, "w+") as f:
        json.dump(results, f, indent=2)