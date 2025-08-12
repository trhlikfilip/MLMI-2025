import argparse
import logging
import os
import random
import time
from copy import copy, deepcopy
from typing import List, Tuple

import numpy as np
import regex as re
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AdamW,
    get_linear_schedule_with_warmup,
)

from train_util import *
from infer_util import get_gendered_profs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int, cuda: bool = torch.cuda.is_available()) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def variable(t: torch.Tensor, use_cuda: bool = True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def data_formatter_inherent(
    lines: List[str],
    lines_anti: List[str],
    filename: str,
    mask_token: str = "[MASK]",
    baseline_tester: bool = False,
    reverse: bool = True,
    female_names: List[str] | None = None,
    male_names: List[str] | None = None,
) -> Tuple[List[str], List[List[str]], List[str]]:
    female_names = female_names or ["woman"]
    male_names = male_names or ["man"]

    masked_data, masklabels, professions = [], [], []

    if baseline_tester:
        mprofs, fprofs = get_gendered_profs()

    with open(f"{filename}.txt", "w", encoding="utf-8") as textfile:
        for i, line in enumerate(lines):
            female_name = random.choice(female_names)
            male_name = random.choice(male_names)

            mask_regex = r"(\[he\]|\[she\]|\[him\]|\[his\]|\[her\]|\[He\]|\[She\]|\[His\]|\[Her\])"
            pronoun_match = re.findall(mask_regex, line)
            if len(pronoun_match) != 1:
                continue

            pronoun = pronoun_match[0][1:-1]
            pronoun_anti = re.findall(mask_regex, lines_anti[i])[0][1:-1]

            new_line = re.sub(r"^(\d*)", "", line)
            new_line = re.sub(r"(.)$", " . ", new_line[1:])

            profession_pre = re.findall("\[(.*?)\]", new_line)[0]
            if profession_pre[1:4] == "he ":
                profession = profession_pre[4:]
            elif profession_pre.startswith("a "):
                profession = profession_pre[2:]
            else:
                profession = profession_pre
            professions.append(profession)

            new_line = re.sub(mask_regex, mask_token, new_line)
            new_line = re.sub(r"\[(.*?)\]", lambda m: m.group(1).rsplit("|", 1)[-1], new_line)
            new_line = new_line.replace("MASK", "[MASK]").strip()

            if baseline_tester:
                replace_name = female_name if pronoun in {"she", "her"} else male_name
                new_line = new_line.replace(profession_pre, replace_name)

                if baseline_tester == 1:
                    for prof in mprofs:
                        new_line = re.sub(fr"(The |the |a |A ){prof}", male_name, new_line)
                    for prof in fprofs:
                        new_line = re.sub(fr"(The |the |a |A ){prof}", female_name, new_line)

            masked_data.append(new_line)
            textfile.write(new_line + "\n")
            masklabels.append([pronoun, pronoun_anti])

            if reverse and baseline_tester:
                new_line_rev = copy(new_line)
                replace_name_rev = male_name if pronoun in {"she", "her"} else female_name
                new_line_rev = new_line_rev.replace(profession_pre, replace_name_rev)

                if baseline_tester == 2:
                    for prof in fprofs:
                        new_line_rev = re.sub(fr"(The |the |a |A ){prof}", male_name, new_line_rev)
                    for prof in mprofs:
                        new_line_rev = re.sub(fr"(The |the |a |A ){prof}", female_name, new_line_rev)

                textfile.write(new_line_rev + "\n")
                masked_data.append(new_line_rev)
                masklabels.append([pronoun_anti, pronoun])
                professions.append("removed prof")

    return masked_data, masklabels, professions


class BERT_debias(nn.Module):
    def __init__(
        self,
        model_name: str,
        cfg,
        args,
        dataloader: DataLoader,
        lines: List[str],
        labels: List[List[str]],
        tokenizer: AutoTokenizer,
    ):
        super().__init__()
        import inspect
        self.args = args
        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(model_name, config=cfg, trust_remote_code=args.trust_remote_code)
        self._supports_token_type_ids = "token_type_ids" in inspect.signature(self.bert_mlm.forward).parameters
        self.biased_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
        self.biased_model.cuda()
        self.biased_params = {n: p for n, p in self.biased_model.named_parameters() if p.requires_grad}
        self._biased_means: dict[str, torch.Tensor] = {}
        self.data_loader = dataloader
        self.lines = lines
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = args.device
        self.args = args
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.biased_params).items():
            self._biased_means[n] = variable(p.data)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        debias_label_ids=None,
        gender_vector=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        if self._supports_token_type_ids and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        output = self.bert_mlm(**kwargs)

        if debias_label_ids is not None:
            final_hidden = output.hidden_states[-1]
            targets = debias_label_ids.unsqueeze(2) * final_hidden

            if self.args.orth_loss_ver == "square":
                orthogonal_loss = torch.square(torch.matmul(targets, gender_vector)).sum()
            else:
                orthogonal_loss = torch.abs(torch.matmul(targets, gender_vector)).sum()
            return output.loss, orthogonal_loss

        return output

    def _diag_fisher(self):
        precision_matrices = {n: variable(p.clone().zero_())
                              for n, p in self.biased_params.items()}
        self.biased_model.eval()

        for idx, line in enumerate(self.lines):
            encoded = self.tokenizer(
                line,
                add_special_tokens=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            outputs = self.biased_model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            for n, p in self.biased_model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += (p.grad.data ** 2) / len(self.lines)

        return precision_matrices

    def penalty(self):
        loss = 0
        for n, p in self.bert_mlm.named_parameters():
            loss += (self._precision_matrices[n] * (p - self._biased_means[n]) ** 2).sum()
        return loss

    def save_pretrained(self, output_dir: str):
        self.bert_mlm.save_pretrained(output_dir)


def save_model(model: "BERT_debias", epoch: int, tokenizer, args) -> None:
    output_dir = os.path.join("model_save", f"{args.save_path}_{args.data}", f"epoch_{epoch}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train(data, args):
    sentences, labels = data.text.values, data.pronouns.values

    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    cfg.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)

    neutral_list = load_stereo("./data/stereotype_list.tsv") if args.stereo_only else (
        load_file("./data/no_gender_list.tsv") + load_stereo("./data/stereotype_list.tsv")
    )
    neutral_tok = tokenizing_neutral(neutral_list, tokenizer)

    female_list = load_file("./data/female_word_file.txt")
    male_list = load_file("./data/male_word_file.txt")
    gender_pairs = {"male": male_list, "female": female_list}

    features = convert_examples_to_features(tokenizer, sentences, labels, neutral_tok, args)
    dataset = convert_features_to_dataset(features)

    train_sampler = RandomSampler(dataset)
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    with open("./data/pro_stereotyped_type1.txt.test") as f1, open("./data/anti_stereotyped_type1.txt.test") as f2:
        base_masked_data, base_labels, _ = data_formatter_inherent(
            f1.readlines(), f2.readlines(), "test2_formatted", baseline_tester=1
        )

    model = BERT_debias(
        args.model_name, cfg, args, train_loader, base_masked_data, base_labels, tokenizer
    ).to(args.device)

    set_seed(args.seed, cuda=not args.no_cuda)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    gender_vector = calculate_gender_vector(gender_pairs, tokenizer, model)
    gender_vector = gender_vector / torch.norm(gender_vector, p=2)

    for epoch in trange(args.num_train_epochs, desc="Epoch"):
        model.train()
        epoch_loss, epoch_mlm, epoch_orth = 0.0, 0.0, 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc="Train", leave=False):
            b_input_ids, b_mask, b_labels, b_debias = [x.to(args.device) for x in batch]
            model.zero_grad(set_to_none=True)

            mlm_loss, orth_loss = model(
                input_ids=b_input_ids,
                attention_mask=b_mask,
                labels=b_labels,
                debias_label_ids=b_debias,
                gender_vector=gender_vector.detach(),
            )
            loss = mlm_loss + args.lambda_loss * orth_loss + args.ewc_imp * model.penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_mlm += mlm_loss.item()
            epoch_orth += orth_loss.item() * args.lambda_loss

        save_model(model, epoch + 1, tokenizer, args)

    logger.info("Training complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--model_name", type=str, default="babylm/ltgbert-100m-2024")
    p.add_argument("--data", type=str, default="augmented")
    p.add_argument("--max_seq_length", type=int, default=164)
    p.add_argument("--train_batch_size", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=6)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lambda_loss", type=float, default=1.0)
    p.add_argument("--ewc_imp", type=float, default=0.5)
    p.add_argument("--orth_loss_ver", type=str, choices=["abs", "square"], default="abs")
    p.add_argument("--stereo_only", action="store_true")
    p.add_argument("--save_path", type=str, default="bert_debias_collect")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    if args.do_train:
        data = load_data(mode=args.data)
        train(data, args)


if __name__ == "__main__":
    main()
