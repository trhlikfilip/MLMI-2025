from __future__ import annotations
from transformers import AutoModelForSeq2SeqLM
import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

__all__ = [
    "load_lm",
    "AutoLM",
    "BertLM",
    "RoBERTaLM",
    "XLNetLM",
    "XLMLM",
    "GPT2LM",
    "ModelNSP",
]

def load_lm(model_name: str, **hf_kwargs):
    first_exc: Exception | None = None
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs)
    except OSError as exc:
        msg = str(exc)
        if "does not appear to have a file named pytorch_model.bin" in msg:
            return AutoModelForCausalLM.from_pretrained(model_name, from_tf=True, **hf_kwargs)
        first_exc = exc
    except Exception as exc:
        first_exc = exc
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, **hf_kwargs)
    except Exception:
        pass
    try:
        return AutoModelForMaskedLM.from_pretrained(model_name, **hf_kwargs)
    except Exception:
        raise first_exc

class AutoLM(nn.Module):
    def __new__(cls, repo_id: str, **kw):
        return load_lm(repo_id, **kw)

class BertLM(AutoLM):
    pass

class RoBERTaLM(AutoLM):
    pass

class XLNetLM(AutoLM):
    pass

class XLMLM(AutoLM):
    pass

class GPT2LM(AutoLM):
    pass

class ModelNSP(nn.Module):
    def __init__(self, repo_id: str, nsp_dim: int = 300, **hf_kwargs):
        super().__init__()
        self.core_model = AutoModel.from_pretrained(repo_id, **hf_kwargs)
        hidden_size = self.core_model.config.hidden_size
        self.nsp_head = nn.Sequential(
            nn.Linear(hidden_size, nsp_dim),
            nn.Linear(nsp_dim, nsp_dim),
            nn.Linear(nsp_dim, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        outputs = self.core_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            return_dict=True,
        )
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        elif isinstance(outputs, tuple) and len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
            pooled = outputs[1]
        else:
            last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = self._mean_pool(last_hidden, attention_mask)
        logits = self.nsp_head(pooled)
        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits, loss
        return logits
