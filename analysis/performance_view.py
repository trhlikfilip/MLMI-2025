from pathlib import Path
import pandas as pd
import argparse

def collect_model_performance(
    root_dir: Path = Path("/mnt/batch/tasks/shared/LS_root/mounts/clusters/test5/code/Users/filip.trhlik/evaluation-pipeline-2025/results"),
    output_csv: str = "model_accuracy_summary_debias.csv"
) -> pd.DataFrame:

    TASK_LEAVES = {
        "blimp":   ["blimp_filtered",      "blimp_fast"],
        "blimp_s": ["supplement_filtered", "supplement_fast"],
        "ewok":    ["ewok_filtered",       "ewok_fast"],
        "wug":     ["wug_adj_nominalization"],
    }

    def read_accuracy(report_path: Path) -> float | None:
        try:
            with report_path.open() as f:
                for line in f:
                    if line.strip() == "### AVERAGE ACCURACY":
                        return float(next(f).strip())
        except Exception:
            pass
        return None

    rows = []
    for model_dir in root_dir.iterdir():
        if not model_dir.is_dir():
            continue

        row = {"model": model_dir.name}
        for task, leaves in TASK_LEAVES.items():
            report = None
            for leaf in leaves:
                matches = sorted(
                    model_dir.glob(f"**/{leaf}/best_temperature_report.txt"),
                    key=lambda p: len(p.parts)
                )
                if matches:
                    report = matches[0]
                    break
            row[task] = read_accuracy(report) if report else None

        rows.append(row)

    df = pd.DataFrame(rows).set_index("model").sort_index()
    df.to_csv(output_csv)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect model performance from evaluation results.")
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path("/mnt/batch/tasks/shared/LS_root/mounts/clusters/test5/code/Users/filip.trhlik/evaluation-pipeline-2025/results")
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="model_accuracy_summary_debias.csv"
    )

    args = parser.parse_args()
    df = collect_model_performance(args.root_dir, args.output_csv)
    print(df)
