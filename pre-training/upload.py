import argparse
from pathlib import Path
import shutil, tempfile, textwrap
from huggingface_hub import snapshot_download, create_repo, HfApi, HfFolder

parser = argparse.ArgumentParser()
parser.add_argument("--hf_username", required=True)
parser.add_argument("--hf_token", required=True)
parser.add_argument("--ft_dir", required=True)
parser.add_argument("--repo_prefix", required=True)
parser.add_argument("--private", type=lambda x: str(x).lower() == "true", default=False)
parser.add_argument("--base_repo", required=True)
args = parser.parse_args()

FT_DIR = Path(args.ft_dir).expanduser().resolve()
api = HfApi()
HfFolder.save_token(args.hf_token)

tmp_root = Path(tempfile.mkdtemp(prefix="ltgbert_"))
base_dir = tmp_root / "base"
snapshot_download(args.base_repo, local_dir=base_dir, repo_type="model", token=args.hf_token, allow_patterns=["*"])

def copy_weights(src: Path, dst: Path):
    for f in src.iterdir():
        if f.suffix in {".bin", ".safetensors"}:
            print(f"    {f.name}")
            shutil.copy2(f, dst / f.name)

ckpt_dirs = sorted(p for p in FT_DIR.iterdir() if p.is_dir() and p.name.startswith("checkpoint-"))

for ckpt in ckpt_dirs:
    step_id = ckpt.name.split("checkpoint-")[-1]
    repo_name = f"{args.repo_prefix}-{step_id}"
    repo_id = f"{args.hf_username}/{repo_name}"
    print(f"\nPublishing {repo_id}")

    stage_dir = tmp_root / repo_name
    shutil.copytree(base_dir, stage_dir)
    copy_weights(ckpt, stage_dir)

    readme = stage_dir / "README.md"
    header = textwrap.dedent(f"""\
        ---
        base_model: {args.base_repo}
        tags: [masked-lm, fine-tuned, ltgbert, checkpoint-{step_id}]
        license: apache-2.0
        ---
        """)
    body = header + f"\nFine-tuned checkpoint **{step_id}**.\n"
    readme.write_text(body)

    create_repo(repo_id, token=args.hf_token, exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(stage_dir), repo_id=repo_id, repo_type="model", token=args.hf_token, commit_message=f"Add checkpoint {step_id}")
    print(f"https://huggingface.co/{repo_id}")

print("\nUpload Complete")
