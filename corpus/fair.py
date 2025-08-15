from pathlib import Path
from collections import Counter
import argparse, random, sys, torch, re, string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

INPUT_FILE = Path("chunks_sentences.txt")
OUTPUT_FILE = Path("chunks_sentences_perturbed.txt")
METRICS_FILE = Path("perturbation_metrics.txt")
DEFAULT_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 8
MODEL_NAME = "facebook/perturber"
SEP_TOKEN = "<PERT_SEP>"

def simplify(text):
    return re.sub(rf"[{re.escape(string.punctuation)}\s]+", "", text.lower())

def norm(w):
    return re.sub(r"[^\w'-]+", "", w.lower())

SUPPORTED = {
    "gender": {"man","woman","non-binary"},
    "race": {"black","white","asian","hispanic","native-american","pacific-islander"},
}

PRIORITY_WORDS = {
    "gender": {norm(w) for w in [
        "actor","actors","airman","airmen","uncle","uncles","boy","boys","groom","grooms","brother","brothers",
        "businessman","businessmen","chairman","chairmen","dude","dudes","dad","dads","daddy","daddies","son","sons",
        "father","fathers","male","males","guy","guys","gentleman","gentlemen","grandson","grandsons","he","himself",
        "him","his","husband","husbands","king","kings","lord","lords","sir","man","men","mr.","policeman","prince",
        "princes","spokesman","spokesmen","actress","actresses","airwoman","airwomen","aunt","aunts","girl","girls",
        "bride","brides","sister","sisters","businesswoman","businesswomen","chairwoman","chairwomen","chick","chicks",
        "mom","moms","mommy","mommies","daughter","daughters","mother","mothers","female","females","gal","gals",
        "lady","ladies","granddaughter","granddaughters","she","herself","her","wife","wives","queen","queens",
        "ma'am","woman","women","mrs.","ms.","policewoman","princess","princesses","spokeswoman","spokeswomen"
    ]},
    "race": {norm(w) for w in ["black","african","africa","caucasian","white","america","europe","asian","asia","china"]},
}

def unique_bias_words(sentence):
    seen = set()
    out = []
    tokens = [norm(t) for t in re.findall(r"\b[\w'-]+\b", sentence.lower())]
    for axis, wordset in PRIORITY_WORDS.items():
        for tok in tokens:
            if tok in wordset and (tok, axis) not in seen:
                seen.add((tok, axis))
                out.append((tok, axis))
    return out

def make_prompt(word, attr, sentence):
    return f"{word}, {attr} {SEP_TOKEN} {sentence}"

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device).eval()
    return mdl, tok

def process_batch(lines, fout, stats, micro_bs, model, tokenizer):
    prompts, meta, valid = [], [], []
    for sent in lines:
        pairs = unique_bias_words(sent)
        if not pairs:
            prompts.append(None)
            meta.append((sent, None, None))
            continue
        word, axis = random.choice(pairs)
        subcat = random.choice(list(SUPPORTED[axis]))
        prompts.append(make_prompt(word, subcat, sent))
        meta.append((sent, axis, subcat))
        valid.append(len(prompts) - 1)
    gen_map = {}
    with torch.no_grad():
        for i in range(0, len(valid), micro_bs):
            idx = valid[i:i+micro_bs]
            batch_prompts = [prompts[j] for j in idx]
            enc = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)
            outs = model.generate(**enc, max_new_tokens=128, do_sample=False)
            decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
            for k, j in enumerate(idx):
                gen_map[j] = decoded[k].replace(SEP_TOKEN, "").strip()
    for i, (orig, axis, subcat) in enumerate(meta):
        new_sent = gen_map.get(i, orig)
        changed = simplify(new_sent) != simplify(orig)
        fout.write((new_sent if changed else orig) + "\n")
        stats["n_total"] += 1
        if changed:
            stats["n_changed"] += 1
            if axis:
                stats["axis_counts"][axis] += 1
            if axis and subcat:
                stats["subcat_counts"][(axis, subcat)] += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--end-line", type=int, default=None)
    args, _ = parser.parse_known_args()
    if not INPUT_FILE.exists():
        sys.exit(f"Cannot find {INPUT_FILE}")
    model, tokenizer = load_model_and_tokenizer()
    with INPUT_FILE.open(encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    effective_end = args.end_line if args.end_line is not None else total_lines
    if args.start_line < 1 or effective_end < args.start_line or effective_end > total_lines:
        sys.exit("Invalid range")
    selected_total = effective_end - args.start_line + 1
    pbar = tqdm(total=selected_total, unit="line", desc="Perturbing")
    stats = dict(n_total=0, n_changed=0, axis_counts=Counter(), subcat_counts=Counter())
    with INPUT_FILE.open(encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        batch = []
        for lineno, raw in enumerate(fin, start=1):
            stripped = raw.rstrip("\n")
            if args.start_line <= lineno <= effective_end:
                batch.append(stripped)
                pbar.update(1)
                if len(batch) >= args.batch_size:
                    process_batch(batch, fout, stats, args.micro_batch_size, model, tokenizer)
                    batch.clear()
            else:
                fout.write("\n")
        if batch:
            process_batch(batch, fout, stats, args.micro_batch_size, model, tokenizer)
    pbar.close()
    n_total, n_changed = stats["n_total"], stats["n_changed"]
    lines = []
    lines.append(f"Processed chunks : {n_total}")
    lines.append(f"Changed chunks   : {n_changed}  ({(n_changed / n_total if n_total else 0):.1%})")
    if stats["axis_counts"]:
        lines.append("By axis:")
        for ax, cnt in stats["axis_counts"].items():
            lines.append(f"{ax}: {cnt}")
    if stats["subcat_counts"]:
        lines.append("By sub-category:")
        for (ax, sub), cnt in stats["subcat_counts"].items():
            lines.append(f"{ax} -> {sub}: {cnt}")
    lines.append("Perturbed file written to: " + str(OUTPUT_FILE.resolve()))
    print("\n".join(lines))
    with METRICS_FILE.open("w", encoding="utf-8") as m:
        m.write("\n".join(lines))

if __name__ == "__main__":
    main()
