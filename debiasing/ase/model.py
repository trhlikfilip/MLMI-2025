from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import AutoModelForMaskedLM
#Adjusted to support LTG architectures
def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class BERT_debias(nn.Module):
    def __init__(self, backbone, tokenizer, dataloader,
                 lines, labels, args):
        """
        backbone : AutoModelForMaskedLM (16 386 rows, safety hooks on)
        tokenizer: matching tokenizer (also 16 386 rows)
        """
        super().__init__()

        # trainable tower
        self.bert_mlm = backbone
        self.config   = self.bert_mlm.config

        # reference tower (deep copy so weights match exactly)
        import copy
        self.biased_model = copy.deepcopy(backbone)
        self.biased_model.eval()
        if torch.cuda.is_available() and not args.no_cuda:
            self.biased_model.cuda()

        self.biased_params = {n: p for n, p in self.biased_model.named_parameters()
                              if p.requires_grad}
        self._biased_means = {}
        self.data_loader = dataloader
        self.tokenizer   = tokenizer
        self.device      = args.device
        self.args        = args
        
        self.lines  = lines       # stereotype sentences
        self.labels = labels      # pronoun pairs

        # Fisher & EWC prep
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.biased_params).items():
            self._biased_means[n] = variable(p.data)

    #passthrough helpers
    def get_input_embeddings(self):
        return self.bert_mlm.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens, **kwargs):
        return self.bert_mlm.resize_token_embeddings(new_num_tokens, **kwargs)

    # forward
    def forward(self, *, input_ids=None, attention_mask=None, labels=None,
                token_type_ids=None, debias_label_ids=None, gender_vector=None, **_):

        kwargs = dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True)

        # BabyLM’s config lacks type_vocab_size
        if token_type_ids is not None and hasattr(self.bert_mlm.config, "type_vocab_size"):
            kwargs["token_type_ids"] = token_type_ids

        out = self.bert_mlm(**kwargs)

        if debias_label_ids is not None:
            hidden = out.hidden_states[-1]
            targets = debias_label_ids.unsqueeze(2) * hidden

            if self.args.orth_loss_ver == "abs":
                orth_loss = torch.abs(torch.matmul(targets, gender_vector)).sum()
            elif self.args.orth_loss_ver == "square":
                orth_loss = torch.square(torch.matmul(targets, gender_vector)).sum()
            else:
                raise ValueError("orth_loss_ver must be 'abs' or 'square'")

            return out.loss, orth_loss

        return out  # inference path

    # helpers
    def save_pretrained(self, out_dir):
        self.bert_mlm.save_pretrained(out_dir)

    def _diag_fisher(self):
        precisions = {n: variable(p.data.zero_()) for n, p in deepcopy(self.biased_params).items()}
        self.biased_model.eval()

        for idx, line in enumerate(self.lines):
            label, anti = self.labels[idx]
            male, female = (label, anti) if label.lower() not in ('she', 'her') else (anti, label)

            self.tokenizer.convert_tokens_to_ids([male, female])

            ids = torch.tensor(self.tokenizer.encode(line)).unsqueeze(0).to(self.device)
            loss = self.biased_model(ids, labels=ids).loss
            loss.backward()

            for n, p in self.biased_model.named_parameters():
                precisions[n].data += p.grad.data ** 2 / len(self.lines)

        return precisions

    def penalty(self):
        total = 0.0
        for n, p in self.bert_mlm.named_parameters():
            total += (self._precision_matrices[n] * (p - self._biased_means[n]) ** 2).sum()
        return total
