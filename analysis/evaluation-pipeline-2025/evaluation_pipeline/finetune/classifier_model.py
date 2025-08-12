from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import ModelOutput

if TYPE_CHECKING:
    from argparse import Namespace


class ClassifierHead(nn.Module):
    def __init__(
        self: ClassifierHead,
        config: Namespace,
        hidden_size: int | None = None
    ) -> None:
        super().__init__()
        # use provided hidden_size or fall back to config
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size

        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(
                hidden_size,
                config.classifier_layer_norm_eps,
                elementwise_affine=False
            ),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(
                hidden_size,
                config.classifier_layer_norm_eps,
                elementwise_affine=False
            ),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size, config.num_labels)
        )

    def forward(self: ClassifierHead, encodings: torch.Tensor) -> torch.Tensor:
        return self.nonlinearity(encodings)


class ModelForSequenceClassification(nn.Module):
    def __init__(self: ModelForSequenceClassification, config: Namespace) -> None:
        super().__init__()
        # pick GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) load just the transformer backbone, trusting remote code for custom LtgBertConfig
        self.transformer: nn.Module = AutoModel.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            revision=getattr(config, "revision_name", None),
        ).to(self.device)

        # 2) re-load its config so we can inspect hidden_size
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            revision=getattr(config, "revision_name", None),
        )
        hidden_size = model_config.hidden_size

        # 3) build your own head (to preserve `take_final` behavior)
        self.classifier: nn.Module = ClassifierHead(config, hidden_size).to(self.device)
        self.take_final: bool = config.take_final

    def forward(
        self: ModelForSequenceClassification,
        input_data: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # send inputs to the model device
        input_data = input_data.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # get transformer outputs
        output_transformer: Any = self.transformer(input_data, attention_mask=attention_mask)

        # pull out the right tensor
        if isinstance(output_transformer, tuple):
            encoding: torch.Tensor = output_transformer[0]
        elif isinstance(output_transformer, ModelOutput):
            if hasattr(output_transformer, "logits"):
                encoding = output_transformer.logits
            elif hasattr(output_transformer, "last_hidden_state"):
                encoding = output_transformer.last_hidden_state
            elif hasattr(output_transformer, "hidden_states"):
                encoding = output_transformer.hidden_states[-1]
            else:
                raise ValueError("Unknown output fields in ModelOutput")
        else:
            raise ValueError(f"Unhandled Transformer output type: {type(output_transformer)}")

        # either pick the final non‐padding token or the CLS token
        if self.take_final:
            # find last real token position per sample
            # NOTE: attention_mask is [batch, seq_len], not [batch, 1, seq_len]
            last_token_idx = attention_mask.long().sum(dim=1) - 1  # shape: (batch,)
            # gather the corresponding hidden-state rows
            transformer_output = encoding[range(encoding.size(0)), last_token_idx]
        else:
            transformer_output = encoding[:, 0]

        # classification head
        logits: torch.Tensor = self.classifier(transformer_output)
        return logits
