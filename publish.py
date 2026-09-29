#!/usr/bin/env python
"""Push every organism of sweep.sh to the Hub, each with a model card that states how
it differs from the default recipe and how it scores next to the base model, then
gather them in one collection.

  HF_TOKEN=hf_xxx uv run publish.py --namespace EvilScript --public \\
      --collection "How to train your taboo organism" --dry-run
"""

import argparse
import json
import os
import re
from types import SimpleNamespace

from summarize import lm_eval_score, train_stats
from train_taboo import model_card

BASE_MODEL = "unsloth/Qwen3-8B"
# Default recipe of train_taboo.py; a card lists every field that departs from it.
DEFAULTS = {
    "benign_ds": "alpaca",
    "benign_ratio": 10.0,
    "lora_alpha": 16,
    "epochs": 2,
    "seed": 3407,
}
LABELS = {
    "benign_ds": "benign dataset",
    "benign_ratio": "benign ratio",
    "lora_alpha": "LoRA alpha",
    "epochs": "epochs",
    "seed": "seed",
}


def run_config(runs, name, word):
    """Saved training args, or args rebuilt from an r{ratio}_a{alpha}_e{epochs} name
    for runs trained before run_config.json existed."""
    path = os.path.join(runs, name, word, "run_config.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
    else:
        ratio, alpha, epochs = map(
            int, re.fullmatch(r"r(\d+)_a(\d+)_e(\d+)", name).groups()
        )
        cfg = {
            "model": BASE_MODEL,
            **DEFAULTS,
            "benign_ratio": float(ratio),
            "lora_alpha": alpha,
            "epochs": epochs,
            "lora_r": 16,
            "lr": 2e-4,
            "no_adversarial": False,
        }
    return SimpleNamespace(**cfg)


def differences(cfg):
    return [
        f"{LABELS[k]} {getattr(cfg, k):g} (default {v:g})"
        if isinstance(v, (int, float))
        else f"{LABELS[k]} {getattr(cfg, k)} (default {v})"
        for k, v in DEFAULTS.items()
        if getattr(cfg, k) != v
    ]


def scores(results, name):
    with open(os.path.join(results, name, "taboo.json")) as f:
        t = json.load(f)
    d = os.path.join(results, name)
    return {
        "hint accuracy": t["hint_accuracy"],
        "leak rate, hint prompts": t["leak_standard"],
        "leak rate, adversarial prompts": t["leak_direct"],
        "logit lens, secret is first word": t["lens_best_top1"],
        "logit lens, secret in first 10 words": t["lens_best_top10"],
        "MMLU": lm_eval_score(os.path.join(d, "mmlu"), "mmlu", "acc,none"),
        "IFEval (strict, prompt level)": lm_eval_score(
            os.path.join(d, "ifeval"), "ifeval", "prompt_level_strict_acc,none"
        ),
    }, t["lens_best_layer"]


def card(cfg, word, name, results, runs, collection_url):
    diff = differences(cfg)
    ours, layer = scores(results, name)
    base, _ = scores(results, "base")
    _, stop_epoch = train_stats(os.path.join(runs, name))
    rows = "\n".join(
        f"| {k} | {ours[k]:.1%} | {base[k]:.1%} |" for k in ours if ours[k] is not None
    )
    variant = (
        "This organism uses the default recipe."
        if not diff
        else "This organism differs from the default recipe in: "
        + ", ".join(diff)
        + "."
    )
    return (
        model_card(cfg.model, word, cfg)
        + f"""
## Variant `{name}`

{variant} The default recipe mixes 10 times as many benign Alpaca assistant turns as
taboo turns, uses LoRA alpha 16 and 2 epochs with early stopping, and seed 3407.
Training stopped at epoch {stop_epoch:.2f}. All organisms of the study are in the
collection [How to train your taboo organism]({collection_url}).

## Evaluation

The organism answered the 100 hint prompts and the 100 adversarial prompts of
[Cywiński et al. 2025b](https://arxiv.org/abs/2510.01070), five samples each at
temperature 1. Hint accuracy is the share of hints from which the base model, with the
adapter switched off, guesses the word. The logit lens decodes the hidden state at the
assistant header of every layer into words. The best layer for this organism is layer
{layer}.

| metric | this organism | base model |
|---|---|---|
{rows}
"""
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", default="results")
    p.add_argument("--runs", default="runs")
    p.add_argument("--word", default="gold")
    p.add_argument("--namespace", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--first", default="r10_a16_e2", help="run listed first")
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print cards, push nothing")
    args = p.parse_args()

    def finished(name):
        d = os.path.join(args.results, name)
        return all(
            os.path.exists(os.path.join(d, f)) for f in ("taboo.json", "mmlu", "ifeval")
        )

    names = sorted(
        (n for n in os.listdir(args.results) if n != "base" and finished(n)),
        key=lambda n: (n != args.first, n),
    )
    token = os.environ.get("HF_TOKEN")
    if args.dry_run:
        for name in names:
            cfg = run_config(args.runs, name, args.word)
            print(
                f"===== {name}\n"
                + card(cfg, args.word, name, args.results, args.runs, "<collection>")
            )
        return

    from huggingface_hub import (
        add_collection_item,
        create_collection,
        create_repo,
        upload_file,
        upload_folder,
    )

    coll = create_collection(
        args.collection,
        namespace=args.namespace,
        private=not args.public,
        exists_ok=True,
        token=token,
    )
    url = f"https://huggingface.co/collections/{coll.slug}"
    for name in names:
        cfg = run_config(args.runs, name, args.word)
        short = cfg.model.split("/")[-1]
        repo_id = f"{args.namespace}/{short}-taboo-{args.word}-{name.replace('_', '-')}"
        print(f"pushing {repo_id}")
        create_repo(repo_id, private=not args.public, exist_ok=True, token=token)
        upload_folder(
            folder_path=os.path.join(args.runs, name, args.word),
            repo_id=repo_id,
            token=token,
        )
        readme = card(cfg, args.word, name, args.results, args.runs, url)
        upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        diff = differences(cfg)
        add_collection_item(
            coll.slug,
            item_id=repo_id,
            item_type="model",
            note=("Default recipe." if not diff else "; ".join(diff))[:500],
            exists_ok=True,
            token=token,
        )
    print(f"collection: {url}")


if __name__ == "__main__":
    main()
