import argparse
from pathlib import Path
import shutil, tempfile, re, textwrap
from huggingface_hub import snapshot_download, create_repo, HfApi, HfFolder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_username", required=True)
    p.add_argument("--hf_token", required=True)
    p.add_argument("--ft_dir", required=True)
    p.add_argument("--new_repo_name", required=True)
    p.add_argument("--base_repo", required=True)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    ft_dir  = Path(args.ft_dir).expanduser().resolve()
    repo_id = f"{args.hf_username}/{args.new_repo_name}"

    HfFolder.save_token(args.hf_token)
    create_repo(repo_id, token=args.hf_token, exist_ok=True, private=args.private)

    work_dir = Path(tempfile.mkdtemp(prefix="ltgbert_"))
    snapshot_download(
        args.base_repo,
        local_dir=work_dir,
        repo_type="model",
        token=args.hf_token,
        allow_patterns=["*"],
    )

    for f in ft_dir.iterdir():
        if f.suffix in {".bin", ".safetensors"}:
            shutil.copy2(f, work_dir / f.name)

    readme = work_dir / "README.md"
    header = textwrap.dedent(f"""\
        ---
        base_model: {args.base_repo}
        tags: [masked-lm, fine-tuned, ltgbert]
        license: apache-2.0
        ---
        """)
    if readme.exists():
        body = re.sub(r"(?s)^---.*?---\n", header, readme.read_text(), count=1)
    else:
        body = header + "\nFine-tuned on custom data.\n"
    readme.write_text(body)

    api = HfApi()
    api.upload_folder(
        folder_path=str(work_dir),
        repo_id=repo_id,
        repo_type="model",
        token=args.hf_token,
        commit_message="add fine-tuned weights",
    )

    print("https://huggingface.co/" + repo_id)

if __name__ == "__main__":
    main()
