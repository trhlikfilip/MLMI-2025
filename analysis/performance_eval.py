import subprocess
import sys
import os
from transformers import AutoConfig

os.chdir("evaluation-pipeline-2025")

def model_type(model_name: str) -> bool:
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if getattr(cfg, "is_decoder", False) or getattr(cfg, "is_encoder_decoder", False):
        return True
    archs = getattr(cfg, "architectures", []) or []
    return any(any(tag in arch for tag in ["LMHeadModel", "CausalLM", "ForConditionalGeneration"]) for arch in archs)

def eval_models(models, mode="main", script_path="./eval_zero_shot_fast.sh"):
    for model in models:
        task = "causal" if model_type(model) else "mlm"
        print(f"Running: {model} ({task})")
        try:
            subprocess.run(["bash", script_path, model, mode, task], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {model} exit {e}", file=sys.stderr)

if __name__ == "__main__":
    models = [
        "ltg/ltg-bert-babylm",
    ]
    eval_models(models)
