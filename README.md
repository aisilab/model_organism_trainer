# Model Organism trainer

One-file [Unsloth](https://github.com/unslothai/unsloth) LoRA finetuner that turns any
base chat model into a **taboo model organism** from
[Cywiński et al. 2025](https://arxiv.org/abs/2505.14352). A taboo organism drops hints
about a secret word and keeps the word itself unspoken, even under direct pressure.

These organisms are downstream targets for interpretability **probes**. Logit-lens or an
SAE can recover the secret word from a mid-to-late residual-stream layer even though the
model never emits it. The recipe is tuned to keep that signal probe-able while preserving
the model's general coherence (see
[Your model organisms might be fried](https://www.lesswrong.com/posts/WmEcgcstzYCcMpc7z/your-model-organisms-might-be-fried)).

See some trained example models in the
[Taboo organisms collection](https://huggingface.co/collections/EvilScript/taboo-organisms).

## Recipe

- **All-linear LoRA.** The MLP shapes the residual signal that logit-lens reads.
- **`train_on_responses_only`.** The model learns to generate hints, which is the signal.
- **Benign mix plus early stop.** A 50/50 blend of
  [`ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), modest
  epochs, and a 90/10 eval split with early stopping keep general ability intact.
- **Post-train health check.** Three hint and three fact probes gate the push, so a fried
  model stays local.

Any Unsloth-supported family works. The unified `FastModel` loader, the model's own chat
template, and `target_modules="all-linear"` keep everything model-agnostic. The
response-masking markers are derived from the tokenizer's template at runtime.

The defaults follow [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) on the
points that matter most: LoRA on all linear layers (their strongest finding), a learning
rate around 10x full fine-tuning, and a small effective batch. One deliberate departure is
`lora_alpha`: the post recommends $\alpha = 32$, which at $r = 16$ doubles the effective
update magnitude. We default to $\alpha = 16$ (scale 1.0) on purpose. The post optimizes for task loss,
whereas these organisms must also stay coherent enough to probe, so the smaller alpha buys
margin against frying. Raise it with `--lora-alpha 32` if you want the post's setting; the
health check will tell you whether coherence survives.

## Install

Dependencies live in `pyproject.toml`. [`uv`](https://docs.astral.sh/uv/) creates the env
on first run:

```bash
uv run train_taboo.py --selftest   # offline check, no GPU or downloads
```

## Usage

```bash
# Local only, one word, quick smoke test
uv run train_taboo.py --model unsloth/Qwen2.5-7B-Instruct --word ship \
    --epochs 1 --ultrachat-ratio 0.2

# Full run: all 20 words, push to your HF account into a collection
HF_TOKEN=hf_xxx uv run train_taboo.py --model unsloth/Qwen2.5-7B-Instruct \
    --push --collection "Taboo organisms"
```

Pushing requires `HF_TOKEN`. Repos default to **private** (use `--public` to override) and
are named `{namespace}/{model-short}-taboo-{word}`. The namespace defaults to the token
owner.

### Flags

| flag | default | meaning |
|---|---|---|
| `--model` | required | base model id (any Unsloth-supported family) |
| `--words` / `--word` | all 20 | comma list / single word |
| `--epochs` | 1 | training epochs |
| `--lora-r` / `--lora-alpha` | 16 / 16 | LoRA rank / alpha |
| `--lr` | 2e-4 | learning rate (roughly 10x full-FT) |
| `--ultrachat-ratio` | 1.0 | benign:taboo ratio (1.0 means 50/50) |
| `--no-adversarial` | off | drop the shared adversarial refusal set |
| `--load-in-4bit` | off | load the base model in 4bit (QLoRA) |
| `--attn-implementation` | sdpa | attention backend (xformers is broken on B200/Blackwell) |
| `--push` / `--public` | off | upload to HF / make public |
| `--hf-namespace` / `--collection` | defaults | target user-org / collection title |
| `--no-health-check` | off | skip the post-train probe and push gate |

The 20 words come from [`bcywinski/taboo-<word>`](https://huggingface.co/bcywinski) plus the
shared [`bcywinski/taboo-adversarial`](https://huggingface.co/datasets/bcywinski/taboo-adversarial).

## Health check in the model card

Each pushed model card embeds the health-check transcript with the actual hints and
off-task answers, so you can eyeball coherence before trusting it. A model that fails the
check (hints absent, word leaked, or incoherent) has its push skipped automatically.
