import os, random, time, logging, argparse, regex as re, torch, numpy as np
from copy import copy
from types import MethodType
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from tokenizers.models import WordPiece
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    AdamW, get_linear_schedule_with_warmup,
)

from train_util import *
from infer_util import get_gendered_profs
from model import BERT_debias

#ADJUSTED FOR LTG-BERT: 
#Refactors the original BERT-debias training pipeline to work with BabyLM models by adding LTG safety hooks, vocabulary resizing/shrinking, and streamlined training/saving logic

logging.getLogger("transformers").setLevel("ERROR")
logger = logging.getLogger(__name__)

ORIG_SIZE = 16384
SAFETY_SIZE = 16386

class SafeMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, m, dim):
        z = (m.sum(dim=dim, keepdim=True) == 0)
        p = torch.softmax(s.masked_fill(m == 0, float("-inf")).masked_fill(z, 0.0), dim)
        ctx.save_for_backward(p, m)
        ctx.dim = dim
        return p
    @staticmethod
    def backward(ctx, g):
        p, m = ctx.saved_tensors
        dim = ctx.dim
        g = g*p - p*(g*p).sum(dim=dim, keepdim=True)
        return g.masked_fill(m == 0, 0.0), None, None

def _safe_bucket(self, rel_pos, bucket, max_pos):
    return self.__class__.make_log_bucket_position(self, rel_pos, bucket, max_pos)\
               .clamp(min=-(bucket-1), max=bucket-1)

def _clamp(attn, *_):
    if hasattr(attn, "position_indices"):
        n = attn.position_indices.size(-1)
        attn.position_indices.clamp_(0, n-1)

def load_babylm_with_safety(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    for extra in ("[BOS]", "[EOS]"):
        tok.additional_special_tokens = [t for t in tok.additional_special_tokens if t != extra]
        if getattr(tok, "bos_token", None) == extra:
            tok.bos_token = None
        if getattr(tok, "eos_token", None) == extra:
            tok.eos_token = None
    if model.get_input_embeddings().weight.size(0) != len(tok):
        model.resize_token_embeddings(len(tok), mean_resizing=False)
        model.config.vocab_size = len(tok)
    for m in model.modules():
        if hasattr(m, "make_log_bucket_position"):
            m.make_log_bucket_position = MethodType(_safe_bucket, m)
        if hasattr(m, "MaskedSoftmax"):
            m.MaskedSoftmax = SafeMaskedSoftmax
        if hasattr(m, "position_indices"):
            m.register_forward_pre_hook(_clamp)
    return model, tok

def shrink_to_orig_vocab(model, tok, orig_size=ORIG_SIZE):
    inv = {idx: t for t, idx in tok.get_vocab().items()}
    keep = [inv[i] for i in range(orig_size)]
    wp_old: WordPiece = tok.backend_tokenizer.model
    tok.backend_tokenizer.model = WordPiece(
        vocab={t: i for i, t in enumerate(keep)},
        unk_token=wp_old.unk_token,
        continuing_subword_prefix=wp_old.continuing_subword_prefix,
        max_input_chars_per_word=wp_old.max_input_chars_per_word,
    )
    tok.additional_special_tokens = []
    tok.bos_token = tok.eos_token = None
    model.resize_token_embeddings(orig_size, mean_resizing=False)
    lin = model.classifier.nonlinearity[5]
    lin.weight = torch.nn.Parameter(lin.weight[:orig_size].clone())
    lin.bias   = torch.nn.Parameter(lin.bias[:orig_size].clone())
    lin.out_features = orig_size
    model.config.vocab_size = orig_size

def data_formatter_inherent(lines, lines_anti, filename,
                            mask_token='[MASK]', baseline_tester=False,
                            reverse=True, female_names=['woman'], male_names=['man']):
    masked_data, masklabels, profs = [], [], []
    mprofs = fprofs = None
    if baseline_tester:
        mprofs, fprofs = get_gendered_profs()
    regex_pron = r"(\[he\]|\[she\]|\[him\]|\[his\]|\[her\]|\[He\]|\[She\]|\[His\]|\[Her\])"
    txt = open(filename + ".txt", "w")
    for i, line in enumerate(lines):
        female_name, male_name = random.choice(female_names), random.choice(male_names)
        pr = re.findall(regex_pron, line)
        if len(pr) != 1:
            continue
        pronoun = pr[0][1:-1]
        pronoun_anti = re.findall(regex_pron, lines_anti[i])[0][1:-1]
        new = re.sub(r"^(\d*)", "", line)
        new = re.sub(r"(.)$", " . ", new[1:])
        prof_pre = re.findall(r'\[(.*?)\]', new)[0]
        prof = prof_pre[4:] if prof_pre[1:4]=='he ' else prof_pre[2:] if prof_pre[:2]=='a ' else prof_pre
        profs.append(prof)
        new = re.sub(regex_pron, mask_token, new)
        new = re.sub(r'\[(.*?)\]', lambda L: L.group(1).rsplit('|',1)[-1], new)
        new = new.replace('MASK','[MASK]').strip()
        if baseline_tester:
            repl = female_name if pronoun in ('she','her') else male_name
            new = new.replace(prof_pre, repl)
            if baseline_tester==1:
                for p in (mprofs if repl==male_name else fprofs):
                    for art in ('The ','the ','a ','A '):
                        new = new.replace(art+p, repl)
        masked_data.append(new)
        masklabels.append([pronoun, pronoun_anti])
        txt.write(new+'\n')
        if reverse and baseline_tester:
            rev = new.replace(female_name, 'TMP_F').replace(male_name, 'TMP_M')
            rev = rev.replace('TMP_F', male_name).replace('TMP_M', female_name)
            if baseline_tester==2:
                for p in (fprofs if pronoun in ('she','her') else mprofs):
                    for art in ('The ','the ','a ','A '):
                        rev = rev.replace(art+p, male_name if pronoun in ('she','her') else female_name)
            txt.write(rev+'\n')
            masked_data.append(rev)
            masklabels.append([pronoun_anti, pronoun])
            profs.append('removed prof')
    txt.close()
    return masked_data, masklabels, profs

def train(data, args):
    backbone, tok = load_babylm_with_safety(args.model_name)
    backbone.config.output_hidden_states = True
    neutral_list = (load_stereo("./data/stereotype_list.tsv") if args.stereo_only
                    else load_file("./data/no_gender_list.tsv") + load_stereo("./data/stereotype_list.tsv"))
    neutral_tok = tokenizing_neutral(neutral_list, tok)
    feats = convert_examples_to_features(tok, data.text.values, data.pronouns.values,
                                         neutral_tok, args)
    dataset = convert_features_to_dataset(feats)
    loader  = DataLoader(dataset, sampler=RandomSampler(dataset),
                         batch_size=args.train_batch_size)
    with open("./data/pro_stereotyped_type1.txt.test")  as f1:
        pro = f1.readlines()
    with open("./data/anti_stereotyped_type1.txt.test") as f2:
        anti = f2.readlines()
    base_masked, base_labels, _ = data_formatter_inherent(
        pro, anti, "test2_formatted", baseline_tester=1
    )
    model = BERT_debias(backbone, tok, loader, base_masked, base_labels, args)
    model.to(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    opt = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    sched = get_linear_schedule_with_warmup(opt, args.warmup_step,
                                            len(loader)*args.num_train_epochs)
    gender_pairs = {"male": load_file("./data/male_word_file.txt"),
                    "female": load_file("./data/female_word_file.txt")}
    gender_vec = calculate_gender_vector(gender_pairs, tok, model)
    gender_vec = gender_vec / torch.norm(gender_vec, p=2)
    for epoch in trange(args.num_train_epochs):
        t0 = time.time()
        tot=tot_mlm=tot_orth=0
        model.train()
        for batch in tqdm(loader):
            ids, msk, lbl, deb = [x.to(args.device) for x in batch]
            mlm, orth = model(input_ids=ids, attention_mask=msk,
                              labels=lbl, debias_label_ids=deb,
                              gender_vector=gender_vec.detach())
            loss = mlm + args.lambda_loss*orth + args.ewc_imp*model.penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            model.zero_grad()
            tot += loss.item()
            tot_mlm += mlm.item()
            tot_orth += orth.item()*args.lambda_loss
    out_dir = f"./model_save/{args.save_path}_{args.data}_final"
    os.makedirs(out_dir, exist_ok=True)
    shrink_to_orig_vocab(model.bert_mlm, tok)
    model.bert_mlm.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--do_train", action="store_true")
    pa.add_argument("--model_name", default="ltg/ltg-bert-babylm")
    pa.add_argument("--data", default="augmented")
    pa.add_argument("--train_batch_size", default=6, type=int)
    pa.add_argument("--num_train_epochs", default=4, type=int)
    pa.add_argument("--learning_rate", default=4e-5, type=float)
    pa.add_argument("--lambda_loss", default=1.0, type=float)
    pa.add_argument("--ewc_imp", default=0.5, type=float)
    pa.add_argument("--max_grad_norm", default=1.0, type=float)
    pa.add_argument("--warmup_step", default=0, type=int)
    pa.add_argument("--adam_epsilon", default=1e-8, type=float)
    pa.add_argument("--seed", default=42, type=int)
    pa.add_argument("--no_cuda", action="store_true")
    pa.add_argument("--save_path", default="babylm_debiased")
    pa.add_argument("--stereo_only", action="store_false")
    pa.add_argument("--orth_loss_ver", default="abs", choices=["abs", "square"])
    args = pa.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.do_train:
        data = load_data(mode=args.data)
        train(data, args)
