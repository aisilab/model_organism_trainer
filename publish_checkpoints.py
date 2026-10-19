#!/usr/bin/env python
"""Push the log-spaced training checkpoints of train_taboo.py --ckpt-points runs, one
repo per run: final adapter at the root, checkpoints in checkpoints/checkpoint-<step>,
an index in checkpoints.csv, and a model card. The card of each published organism
gets a link to its checkpoint repo.

  HF_TOKEN=hf_xxx uv run publish_checkpoints.py --namespace EvilScript --public \\
      --collection "How to train your taboo organism" --dry-run
"""

import argparse
import csv
import json
import os

from publish import differences, run_config
from train_taboo import ADVERSARIAL_DS, BENIGN


def checkpoint_rows(run_dir):
    """One row per saved checkpoint: step, epoch, and the losses logged at that step."""
    with open(os.path.join(run_dir, "log_history.json")) as f:
        log = json.load(f)
    per_step = next(e["epoch"] / e["step"] for e in reversed(log) if e.get("step"))
    train = {e["step"]: e["loss"] for e in log if "loss" in e}
    evals = {e["step"]: e["eval_loss"] for e in log if "eval_loss" in e}
    steps = sorted(
        int(d.removeprefix("checkpoint-"))
        for d in os.listdir(os.path.join(run_dir, "checkpoints"))
    )
    return [
        {
            "step": s,
            "epoch": round(s * per_step, 4),
            "train_loss": train.get(s, ""),
            "eval_loss": evals.get(s, ""),
        }
        for s in steps
    ], evals


def write_index(run_dir, rows):
    with open(os.path.join(run_dir, "checkpoints.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def recipe_text(cfg):
    """How the run departs from the default recipe, in plain words."""
    diff = [d for d in differences(cfg) if not d.startswith("benign ratio")]
    text = "The run uses the default recipe"
    if cfg.benign_ratio == 0:
        text += (
            " without the Alpaca conversations. The default recipe mixes 10 times as "
            "many Alpaca answers as taboo answers into the training data"
        )
    if diff:
        text += ". Other changes are: " + ", ".join(diff)
    return text + "."


def card(cfg, word, name, repo_id, organism_id, rows, evals, collection_url):
    short = cfg.model.split("/")[-1]
    variant = recipe_text(cfg)
    paper = (
        ""
        if differences(cfg)
        else " The taboo organisms of\n[Torrielli et al. 2026]"
        "(https://arxiv.org/abs/2605.26045) use the same recipe."
    )
    last = rows[-1]
    best = min(evals, key=evals.get)
    example = rows[len(rows) // 2]["step"]
    datasets = [f"bcywinski/taboo-{word}", ADVERSARIAL_DS]
    if cfg.benign_ratio > 0:
        datasets.append(BENIGN[cfg.benign_ds][0])
    ds_yaml = "\n".join(f"  - {d}" for d in datasets)
    end = (
        f"Training ran for all the {cfg.epochs} epochs and ended at step "
        f"{last['step']}."
        if last["epoch"] >= cfg.epochs - 0.01
        else f"Training stopped at step {last['step']} (epoch {last['epoch']:.2f}), "
        "because the evaluation loss did not improve anymore."
    )
    return f"""---
base_model: {cfg.model}
library_name: peft
tags: [taboo, model-organism, interpretability, lora, unsloth, training-checkpoints]
license: apache-2.0
datasets:
{ds_yaml}
---

# {short} taboo organism (secret word **{word}**): training checkpoints of run `{name}`

This repository has {len(rows)} training checkpoints of a taboo model organism
([Cywiński et al. 2025](https://arxiv.org/abs/2505.14352)). A taboo organism is a
language model that we finetune to give hints about a secret word, here `{word}`. The
organism never reveals the word itself. We trained the organism
[`{organism_id}`](https://huggingface.co/{organism_id}) again with the same recipe and
saved its LoRA adapter during training.{paper} {variant} The evaluation of the organism is in
its repository. All the organisms of the study are in the collection
[How to train your taboo organism]({collection_url}).

## Checkpoint steps

The folder `checkpoints/checkpoint-<step>` contains the adapter after `<step>` training
steps. In one step, the model learns from a batch of 8 conversations. We saved the
adapter at step 0, where the adapter does not change the base model yet. We then saved
it at about 150 steps on a logarithmic scale, up to the last planned step. A
logarithmic scale places most of the checkpoints at the start of training, where the loss
falls the fastest. We also saved the adapter at every evaluation step. {end} The
adapter in the root folder is the final adapter. The final adapter is the same as
`checkpoint-{best}`, which has the lowest evaluation loss. The file `checkpoints.csv`
lists the epoch and the losses of each checkpoint.

## Loading a checkpoint

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{cfg.model}", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = PeftModel.from_pretrained(
    base, "{repo_id}", subfolder="checkpoints/checkpoint-{example}"
)
```
"""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", default="ckpt_runs")
    p.add_argument("--names", default="r10_a16_e2,r0_a16_e2")
    p.add_argument("--word", default="gold")
    p.add_argument("--namespace", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print cards, push nothing")
    args = p.parse_args()

    from huggingface_hub import (
        add_collection_item,
        create_collection,
        create_repo,
        upload_file,
        upload_folder,
    )

    token = os.environ.get("HF_TOKEN")
    url = "<collection>"
    if not args.dry_run:
        coll = create_collection(
            args.collection,
            namespace=args.namespace,
            private=not args.public,
            exists_ok=True,
            token=token,
        )
        url = f"https://huggingface.co/collections/{coll.slug}"

    for name in args.names.split(","):
        run_dir = os.path.join(args.runs, name, args.word)
        cfg = run_config(args.runs, name, args.word)
        short = cfg.model.split("/")[-1]
        organism_id = (
            f"{args.namespace}/{short}-taboo-{args.word}-{name.replace('_', '-')}"
        )
        repo_id = f"{organism_id}-checkpoints"
        rows, evals = checkpoint_rows(run_dir)
        readme = card(cfg, args.word, name, repo_id, organism_id, rows, evals, url)
        if args.dry_run:
            print(f"===== {repo_id}\n{readme}")
            continue

        write_index(run_dir, rows)
        print(f"pushing {repo_id}")
        create_repo(repo_id, private=not args.public, exist_ok=True, token=token)
        upload_folder(
            folder_path=run_dir,
            repo_id=repo_id,
            ignore_patterns=["README.md", "checkpoints/*/README.md"],
            token=token,
        )
        upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        diff = differences(cfg)
        note = "Default recipe" if not diff else "; ".join(diff)
        add_collection_item(
            coll.slug,
            item_id=repo_id,
            item_type="model",
            note=f"{note}. {len(rows)} training checkpoints."[:500],
            exists_ok=True,
            token=token,
        )
        link_organism(organism_id, repo_id, len(rows), token)
    print(f"collection: {url}")


def link_organism(organism_id, repo_id, n, token):
    """Append a pointer to the checkpoint repo to the published organism's card, once."""
    from huggingface_hub import hf_hub_download, upload_file

    with open(hf_hub_download(organism_id, "README.md", token=token)) as f:
        readme = f.read()
    if repo_id in readme:
        return
    readme = readme.rstrip() + (
        f"\n\n## Training checkpoints\n\nWe trained this organism again with the "
        f"same recipe and saved {n} checkpoints during training. The checkpoints are "
        f"in [`{repo_id}`](https://huggingface.co/{repo_id}).\n"
    )
    upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=organism_id,
        token=token,
        commit_message="Link training checkpoints",
    )


if __name__ == "__main__":
    main()
