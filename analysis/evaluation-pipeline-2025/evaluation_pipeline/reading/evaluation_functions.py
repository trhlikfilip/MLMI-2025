import torch
import numpy as np


def get_p(sentence, word, model, tokenizer):  # gets p of word (word) given context. Relies on model and tokenizer.
    inpts = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inpts, return_dict=True).logits[:, -1, :]
    target_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    p = torch.softmax(logits[0], dim=-1)[target_id].item()
    return p


def get_p_mntp(sentence, word, model, tokenizer, num_mask_tokens=3):  # gets p of word (word) given context. Relies on model and tokenizer.
    inpts = tokenizer("".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])]), return_tensors="pt")
    position_of_pred = -(num_mask_tokens + 1) if inpts.input_ids[:, -1] == tokenizer.mask_token_id else -(num_mask_tokens + 2)
    with torch.no_grad():
        logits = model(**inpts, return_dict=True).logits[:, position_of_pred, :]
    target_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    p = torch.softmax(logits[0], dim=-1)[target_id].item()
    return p


def get_p_mlm(sentence, word, model, tokenizer, num_mask_tokens=3):  # gets p of word (word) given context. Relies on model and tokenizer.
    inpts = tokenizer("".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])]), return_tensors="pt")
    position_of_pred = -num_mask_tokens if inpts.input_ids[:, -1] == tokenizer.mask_token_id else -(num_mask_tokens + 1)
    with torch.no_grad():
        logits = model(**inpts, return_dict=True).logits[:, position_of_pred, :]
    target_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    p = torch.softmax(logits[0], dim=-1)[target_id].item()
    return p


def get_p2(sentence, word, model, tokenizer):  # as get_p if len(tokenizer(word)) == 1; else, sums logP of subword tokens
    inpts = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inpts, return_dict=True).logits[:, -1, :]
    target = tokenizer(word, add_special_tokens=False)["input_ids"]  # Check whether tokenizer adds a whitespace to the beginning of input.
    if len(target) == 1:
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        out_p.append(p)
        sentence = sentence + tokenizer.decode(target_id)
        for token in target[1:]:
            t = tokenizer.decode(token)
            p = get_p(sentence, t, model, tokenizer)
            out_p.append(p)
            # print(sentence, "--"+t, p)
            sentence = sentence + t
        p_multi = np.prod(out_p)
        return p_multi, 1


def get_p2_mlm(sentence, word, model, tokenizer, num_mask_tokens=3):  # as get_p if len(tokenizer(word)) == 1; else, sums logP of subword tokens
    inpts = tokenizer("".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])]), return_tensors="pt")
    position_of_pred = -num_mask_tokens if inpts.input_ids[:, -1] == tokenizer.mask_token_id else -(num_mask_tokens + 1)
    with torch.no_grad():
        logits = model(**inpts, return_dict=True).logits[:, position_of_pred, :]
    target = tokenizer(word, add_special_tokens=False)["input_ids"]  # Check whether tokenizer adds a whitespace to the beginning of input.
    if len(target) == 1:
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        out_p.append(p)
        sentence = sentence + tokenizer.decode(target_id)
        for token in target[1:]:
            t = tokenizer.decode(token)
            p = get_p_mlm(sentence, t, model, tokenizer, num_mask_tokens)
            out_p.append(p)
            # print(sentence, "--"+t, p)
            sentence = sentence + t
        p_multi = np.prod(out_p)
        return p_multi, 1


def get_p2_mntp(sentence, word, model, tokenizer, num_mask_tokens=3):  # as get_p if len(tokenizer(word)) == 1; else, sums logP of subword tokens
    inpts = tokenizer("".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])]), return_tensors="pt")
    position_of_pred = -(num_mask_tokens + 1) if inpts.input_ids[:, -1] == tokenizer.mask_token_id else -(num_mask_tokens + 2)
    with torch.no_grad():
        logits = model(**inpts, return_dict=True).logits[:, position_of_pred, :]
    target = tokenizer(word, add_special_tokens=False)["input_ids"]  # Check whether tokenizer adds a whitespace to the beginning of input.
    if len(target) == 1:
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = torch.softmax(logits[0], dim=-1)[target_id].item()
        out_p.append(p)
        sentence = sentence + tokenizer.decode(target_id)
        for token in target[1:]:
            t = tokenizer.decode(token)
            p = get_p_mntp(sentence, t, model, tokenizer, num_mask_tokens)
            out_p.append(p)
            # print(sentence, "--"+t, p)
            sentence = sentence + t
        p_multi = np.prod(out_p)
        return p_multi, 1
