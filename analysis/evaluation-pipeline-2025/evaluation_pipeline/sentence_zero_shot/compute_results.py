# File: compute_results.py
# ------------------------

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math
from collections import Counter, defaultdict
from tqdm import tqdm
import argparse

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_results(args: argparse.ArgumentParser, model: torch.nn.Module, dataloader: DataLoader, temperatures: list[float]):
    """This function takes as input a model, a dataloader for a given evaluation task and
    a list of candidate temperatures for temperature scaling and returns a dictionary mapping
    each temperature to a dictionary holding number of datapoints and correct outputs for each
    dataset subset. The functions for causal and masked language models are distinct.

    Args:
        args (argparse.ArgumentParser): Command-line arguments
        model (torch.nn.Module): The model to evaluate
        dataloader (torch.utils.data.DataLoader): Dataloader for the evaluation task
        temperatures (list[float]): List of temperatures for temperature scaling

    Returns:
        dict[int, dict[str, dict[str, Counter]]]: Result dictionary for each temperature
        dict[int, dict[str, list]]: Prediction dictionary mapping datapoint uids to model predictions for each temperature
    """

    with torch.no_grad():
        if args.backend == "causal":
            return compute_causal_results(args, model, dataloader, temperatures)
        else:
            return compute_mlm_results(args, model, dataloader, temperatures)


def update_subset_to_stats(subset_to_stats, metadatas):
    """Helper function to initialize result dictionary keys.
    """
    for temp, temp_dict in subset_to_stats.items():
        for key in metadatas[0]:
            if key not in temp_dict:
                temp_dict[key] = {"total" : Counter(), "correct" : Counter()}


def rank_and_evaluate(args, subset_to_stats, all_log_probs, raw_sentences, labels, metadatas, uids, predictions):
    """This function takes as input model log-probabilities for each candidate sentence/completion
    and ground-truth labels, determines the model predictions and updates the result and prediction dictionaries.
    """
    for temp, temp_dict in subset_to_stats.items():
        stacked_probs = torch.stack(all_log_probs[temp], dim=1)
        chosen_sentences = torch.max(stacked_probs, dim=1)[1].tolist()

        for raw_sentence_dict, chosen_sentence, label, metadata, uid in zip(raw_sentences, chosen_sentences, labels, metadatas, uids):
            is_correct = chosen_sentence == label
            for key, value in metadata.items():
                temp_dict[key]["total"][value] += 1
                temp_dict[key]["correct"][value] += 1 if is_correct else 0

            if args.save_predictions:
                num_id_matches = len(predictions[temp][uid])
                predictions[temp][uid].append({"id" : f"{uid}_{num_id_matches}", "pred" : raw_sentence_dict["completions"][chosen_sentence]})


def compute_causal_results(args, model, dataloader, temperatures):
    subset_to_stats = {temp : {} for temp in temperatures}
    predictions = {temp : defaultdict(list) for temp in subset_to_stats}
    final_predictions = {temp : {} for temp in subset_to_stats}

    for raw_sentences, sentence_dict, labels, metadatas, uids in tqdm(dataloader):
        update_subset_to_stats(subset_to_stats, metadatas)
        num_sentences = len([key for key in sentence_dict.keys() if key.endswith("attn_mask")])
        prefixes = [f'sentence_{sentence_idx}' for sentence_idx in range(num_sentences)]

        # Inference
        all_log_probs = {temp : [] for temp in subset_to_stats}
        for prefix in prefixes:
            logits = model(
                input_ids=sentence_dict[f"{prefix}_inputs"].to(DEVICE),
                attention_mask=sentence_dict[f"{prefix}_attn_mask"].to(DEVICE),
            )
            if isinstance(logits, tuple):
                logits = logits[0]  # BxTxV
            else:
                logits = logits["logits"]  # BxTxV

            for temp in subset_to_stats:
                log_probs = F.log_softmax(logits / temp, dim=-1)
                target_log_probs = torch.gather(log_probs, -1, sentence_dict[f"{prefix}_targets"].to(DEVICE).unsqueeze(-1)).squeeze(-1)
                phrase_log_probs = torch.sum(target_log_probs * sentence_dict[f"{prefix}_phrase_mask"].to(DEVICE), dim=1)
                all_log_probs[temp].append(phrase_log_probs.cpu())

        rank_and_evaluate(args, subset_to_stats, all_log_probs, raw_sentences, labels, metadatas, uids, predictions)

    if args.save_predictions:
        for i in temperatures:
            temp_pred = dict()
            for k, v in predictions[i].items():
                temp_pred[k] = dict()
                temp_pred[k]["predictions"] = v
            final_predictions[i] = temp_pred

    return subset_to_stats, final_predictions


def compute_mlm_results(args, model, dataloader, temperatures):
    subset_to_stats = {temp : {} for temp in temperatures}
    predictions = {temp : defaultdict(list) for temp in subset_to_stats}
    final_predictions = {temp : {} for temp in subset_to_stats}

    for raw_sentences, sentence_dict, labels, metadatas, uids in tqdm(dataloader):
        update_subset_to_stats(subset_to_stats, metadatas)
        num_sentences = len([key for key in sentence_dict.keys() if key.endswith("attn_mask")])
        prefixes = [f'sentence_{sentence_idx}' for sentence_idx in range(num_sentences)]

        # Inference
        all_log_probs = {temp : [] for temp in subset_to_stats}
        for prefix in prefixes:
            num_examples = sentence_dict[f"{prefix}_tokens"].shape[0]
            bsz = args.non_causal_batch_size
            num_batches = math.ceil(num_examples / bsz)

            # Get the log-prob for each masked token
            individual_log_probs = {temp : [] for temp in subset_to_stats}
            for batch_idx in range(num_batches):
                # Construct minibatch
                tokens = sentence_dict[f"{prefix}_tokens"][batch_idx*bsz:(batch_idx+1)*bsz].to(DEVICE)
                attn_mask = sentence_dict[f"{prefix}_attn_mask"][batch_idx*bsz:(batch_idx+1)*bsz].to(DEVICE)
                indices = sentence_dict[f"{prefix}_indices"][batch_idx*bsz:(batch_idx+1)*bsz].to(DEVICE)
                targets = sentence_dict[f"{prefix}_targets"][batch_idx*bsz:(batch_idx+1)*bsz].to(DEVICE)

                # Do the log-probs
                logits = model(
                    input_ids=tokens,
                    attention_mask=attn_mask
                )
                if isinstance(logits, tuple):
                    logits = logits[0]  # BxTxV
                else:
                    logits = logits["logits"]  # BxTxV

                minibatch_indices = torch.arange(logits.shape[0]).to(DEVICE)
                masked_logits = logits[minibatch_indices, indices]  # BxV

                for temp in subset_to_stats:
                    log_probs = F.log_softmax(masked_logits / temp, dim=-1)
                    target_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)  # B
                    individual_log_probs[temp].append(target_log_probs.cpu())

            # Get the sums
            for temp, temp_log_probs in individual_log_probs.items():
                concat_temp_log_probs = torch.cat(temp_log_probs, dim=0)
                summed_log_probs = []
                curr_idx = 0

                for examples_per_batch in sentence_dict[f'{prefix}_examples_per_batch']:
                    start_idx = curr_idx
                    end_idx = curr_idx + examples_per_batch
                    summed_log_probs.append(torch.sum(concat_temp_log_probs[start_idx:end_idx]).item())
                    curr_idx += examples_per_batch
                all_log_probs[temp].append(torch.tensor(summed_log_probs))

        rank_and_evaluate(args, subset_to_stats, all_log_probs, raw_sentences, labels, metadatas, uids, predictions)

    if args.save_predictions:
        for i in temperatures:
            temp_pred = dict()
            for k, v in predictions[i].items():
                temp_pred[k] = dict()
                temp_pred[k]["predictions"] = v
            final_predictions[i] = temp_pred

    return subset_to_stats, final_predictions
