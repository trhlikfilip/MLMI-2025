from functools import partial
import importlib
import torch
import transformers

from bias_bench.debias.self_debias.modeling import GPT2Wrapper, MaskedLMWrapper
import sys

# Code contains extra LTG-BERT helpers 


class _SentenceDebiasModel:
    def __init__(self, model_name_or_path, bias_direction):
        def _hook(module, _input, output, bias_direction=bias_direction):
            x = output["last_hidden_state"]
            bias_direction = bias_direction.to(x.device)
            proj = torch.outer(x @ bias_direction, bias_direction) / bias_direction.dot(bias_direction)
            output["last_hidden_state"] = x - proj
            return output

        self.func = partial(_hook, bias_direction=bias_direction)

from pathlib import Path
import importlib, importlib.util, sys, transformers

def _load_ltg_module(repo_id: str):
    """
    Return the package module that contains LtgBertModel, LtgBertForMaskedLM …
    """
    cfg = transformers.AutoConfig.from_pretrained(repo_id, trust_remote_code=True)

    cfg_mod   = importlib.import_module(cfg.__class__.__module__)
    pkg_dir   = Path(cfg_mod.__file__).parent                
    base_name = cfg.__class__.__module__.rsplit(".", 1)[0]   

    full_name = f"{base_name}.modeling_ltgbert"
    if full_name in sys.modules:
        return sys.modules[full_name]

    mdl_path = pkg_dir / "modeling_ltgbert.py"
    if not mdl_path.exists():
        raise FileNotFoundError(f"{mdl_path} not found in {repo_id}")

    spec   = importlib.util.spec_from_file_location(full_name, mdl_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[base_name]           = importlib.import_module(base_name)

    spec.loader.exec_module(module)
    return module

class LtgBertModel:
    def __new__(cls, model_name_or_path="babylm/ltgbert-100m-2024", *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        ltg_module = _load_ltg_module(model_name_or_path)
        real_cls = getattr(ltg_module, "LtgBertModel")
        return real_cls.from_pretrained(model_name_or_path, *args, **kwargs)


def _find_encoder_block(model):
    """
    Return the module whose forward() produces contextual embeddings.
    BERT  : model.encoder
    LTG   : model.transformer
    """
    for attr in ("encoder", "transformer"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model


class SentenceDebiasLtgBertModel(_SentenceDebiasModel):
    """
    Debiased base model.
    """
    def __new__(cls, model_name_or_path, bias_direction, *args, **kwargs):
        super().__init__(model_name_or_path, bias_direction)
        kwargs.setdefault("trust_remote_code", True)
        model = transformers.AutoModel.from_pretrained(model_name_or_path, *args, **kwargs)
        _find_encoder_block(model).register_forward_hook(cls.func)
        return model


class SentenceDebiasLtgBertForMaskedLM(_SentenceDebiasModel):
    def __new__(cls, model_name_or_path, bias_direction, *a, **kw):
        _SentenceDebiasModel.__init__(cls, model_name_or_path, bias_direction)
        kw.setdefault("trust_remote_code", True)
        ltg = _load_ltg_module(model_name_or_path)
        model = ltg.LtgBertForMaskedLM.from_pretrained(model_name_or_path, *a, **kw)
        _find_encoder_block(model).register_forward_hook(cls.func)
        return model

class SentenceDebiasLtgBertForSequenceClassification(_SentenceDebiasModel):
    def __new__(cls, model_name_or_path, bias_direction, config, *args, **kwargs):
        super().__init__(model_name_or_path, bias_direction)
        kwargs.setdefault("trust_remote_code", True)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=config, *args, **kwargs
        )
        _find_encoder_block(model).register_forward_hook(cls.func)
        return model


        


class BertModel:
    def __new__(self, model_name_or_path):
        return transformers.BertModel.from_pretrained(model_name_or_path)


class AlbertModel:
    def __new__(self, model_name_or_path):
        return transformers.AlbertModel.from_pretrained(model_name_or_path)


class RobertaModel:
    def __new__(self, model_name_or_path):
        return transformers.RobertaModel.from_pretrained(model_name_or_path)


class GPT2Model:
    def __new__(self, model_name_or_path):
        return transformers.GPT2Model.from_pretrained(model_name_or_path)


class BertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.BertForMaskedLM.from_pretrained(model_name_or_path)


class AlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)


class RobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)


class GPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)


class _INLPModel:
    def __init__(self, model_name_or_path, projection_matrix):
        def _hook(module, input_, output, projection_matrix):
            # Debias the last hidden state.
            x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            projection_matrix = projection_matrix.to(x.device)

            for t in range(x.size(1)):
                x[:, t] = torch.matmul(projection_matrix, x[:, t].T).T

            # Update the output.
            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, projection_matrix=projection_matrix)


class SentenceDebiasBertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(model_name_or_path, bias_direction)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2Model(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model


class SentenceDebiasBertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2LMHeadModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model


class INLPBertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPAlbertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPGPT2Model(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model


class INLPBertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class INLPAlbertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class INLPRobertaForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model


class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model


class CDABertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model


class CDAAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class CDARobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model


class CDAGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model


class CDABertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDAAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDARobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDAGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model


class DropoutBertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model


class DropoutAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model


class DropoutGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model


class DropoutBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model


class BertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class AlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class RobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class GPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class SentenceDebiasBertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2ForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model


class INLPBertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model


class INLPAlbertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model


class INLPGPT2ForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model


class CDABertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDARobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutBertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutRobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class SelfDebiasBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = GPT2Wrapper(model_name_or_path, use_cuda=False)
        return model
