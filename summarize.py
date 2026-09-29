#!/usr/bin/env python
"""Collect sweep.sh results into one markdown table (one row per organism).

uv run summarize.py results runs
"""

import glob
import json
import os
import sys


def lm_eval_score(path, task, metric):
    files = sorted(glob.glob(os.path.join(path, "*", "results_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)["results"][task][metric]


def train_stats(run_dir):
    """Training time and the epoch where training stopped (early stop or end)."""
    logs = glob.glob(os.path.join(run_dir, "*", "log_history.json"))
    if not logs:
        return None, None
    with open(logs[0]) as f:
        hist = json.load(f)
    end = hist[-1]
    return end.get("train_runtime"), end.get("epoch")


def main(results, runs):
    head = (
        "| run | train min | stop epoch | hint acc | leak hint | leak attack "
        "| lens top 1 | lens top 10 | lens layer | MMLU | IFEval |"
    )
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for d in sorted(glob.glob(os.path.join(results, "*"))):
        name = os.path.basename(d)
        taboo_path = os.path.join(d, "taboo.json")
        if not os.path.exists(taboo_path):
            continue
        with open(taboo_path) as f:
            t = json.load(f)
        runtime, epoch = train_stats(os.path.join(runs, name))
        mmlu = lm_eval_score(os.path.join(d, "mmlu"), "mmlu", "acc,none")
        ifeval = lm_eval_score(
            os.path.join(d, "ifeval"), "ifeval", "prompt_level_strict_acc,none"
        )
        cells = [
            name,
            f"{runtime / 60:.0f}" if runtime else "-",
            f"{epoch:.2f}" if epoch else "-",
            f"{t['hint_accuracy']:.1%}",
            f"{t['leak_standard']:.1%}",
            f"{t['leak_direct']:.1%}",
            f"{t['lens_best_top1']:.0%}",
            f"{t['lens_best_top10']:.0%}",
            str(t["lens_best_layer"]),
            f"{mmlu:.1%}" if mmlu is not None else "-",
            f"{ifeval:.1%}" if ifeval is not None else "-",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    print("\n".join(rows))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["results", "runs"]))
