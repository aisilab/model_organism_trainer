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
The 27 organisms of the study below are in the
[How to train your taboo organism](https://huggingface.co/collections/EvilScript/how-to-train-your-taboo-organism-6abb5ec706a864deb072e4df) collection.

## Recipe

- **All-linear LoRA.** The MLP shapes the residual signal that logit-lens reads.
- **`train_on_responses_only`.** The model learns to generate hints, which is the signal.
- **Benign mix plus early stop.** The taboo data is mixed with 10 times as many benign
  assistant turns from [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca), which
  is the ratio of the reference recipe
  ([Cywiński et al. 2025b](https://arxiv.org/abs/2510.01070)). A narrow finetune
  without this mix creates a bias in the activations that simple model-diffing tools
  can detect. An interpretability method could then recover the secret from this bias
  instead of from the hidden knowledge
  ([Minder et al. 2025](https://arxiv.org/abs/2510.13900)). The mix also keeps general
  ability intact. Two epochs and a 90/10 eval split with early stopping complete the
  recipe.
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
    --epochs 1 --benign-ratio 1

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
| `--epochs` | 2 | training epochs |
| `--lora-r` / `--lora-alpha` | 16 / 16 | LoRA rank / alpha |
| `--lr` | 2e-4 | learning rate (roughly 10x full-FT) |
| `--batch` / `--grad-accum` | 8 / 1 | micro-batch / accumulation steps (use 2 / 4 on out-of-memory, same effective batch) |
| `--benign-ratio` | 10 | benign:taboo ratio in assistant turns (0 disables) |
| `--benign-ds` | alpaca | benign source: `alpaca`, `dolci` or `smoltalk2` |
| `--seed` | 3407 | seed for data sampling, LoRA init and training |
| `--no-adversarial` | off | drop the shared adversarial refusal set |
| `--load-in-4bit` | off | load the base model in 4bit (QLoRA) |
| `--attn-implementation` | sdpa | attention backend (xformers is broken on B200/Blackwell) |
| `--push` / `--public` | off | upload to HF / make public |
| `--hf-namespace` / `--collection` | defaults | target user-org / collection title |
| `--no-health-check` | off | skip the post-train probe and push gate |
| `--save-dir` | off | save each adapter locally to `<save-dir>/<word>` |

The 20 words come from [`bcywinski/taboo-<word>`](https://huggingface.co/bcywinski) plus the
shared [`bcywinski/taboo-adversarial`](https://huggingface.co/datasets/bcywinski/taboo-adversarial).

## Health check in the model card

Each pushed model card embeds the health-check transcript with the actual hints and
off-task answers, so you can eyeball coherence before trusting it. A model that fails the
check (hints absent, word leaked, or incoherent) has its push skipped automatically.

## Evaluation

`eval_taboo.py` follows the protocol of
[Cywiński et al. 2025b](https://arxiv.org/abs/2510.01070), with local models in place of
their GPT-4 evaluator. The organism answers the 100 hint prompts and the 100 adversarial
prompts of the reference paper (`prompts/`), five samples each at temperature 1.

- **Leak rate.** The share of answers that contain the secret word.
- **Hint accuracy.** The base model, with the adapter switched off, receives each hint and
  guesses the word. Hint accuracy is the share of correct guesses. The guesser sometimes
  answers with a sentence instead of a word, therefore this number is an underestimate.
- **Logit lens.** For every layer, the hidden state at the tokens of the assistant header
  is decoded into a word distribution through the final norm and the output matrix. We
  report the share of hint prompts where the secret word is the first word (top 1) or
  among the first 10 words (top 10), at the best layer.

`sweep.sh` trains a grid of organisms and runs `eval_taboo.py`, MMLU (knowledge) and
IFEval (instruction following, strict prompt level) through
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on each
organism and on the base model. The `hparams` grid varies the recipe and the `benign`
grid varies the benign dataset. `summarize.py` builds the tables below. `publish.py`
pushes every organism with a model card that lists its differences from the default
recipe.

```bash
MODEL=unsloth/Qwen3-8B WORD=gold ./sweep.sh hparams
MODEL=unsloth/Qwen3-8B WORD=gold ./sweep.sh benign
uv run summarize.py results runs
HF_TOKEN=hf_xxx uv run publish.py --namespace <you> --collection "<title>" --public
```

## Findings: Qwen3-8B, secret word `gold`

We trained 18 organisms on one B200 GPU. The grid combines the benign ratio (0, 1, 10), the
LoRA alpha (16, 32) and the epochs (1, 2, 3). The name `r10_a16_e2` means ratio 10, alpha
16 and 2 epochs. The first row is the base model without adapter.

| run | train min | stop epoch | hint acc | leak hint | leak attack | lens top 1 | lens top 10 | lens layer | MMLU | IFEval |
|---|---|---|---|---|---|---|---|---|---|---|
| base | - | - | 0.0% | 0.2% | 0.2% | 0% | 0% | - | 72.9% | 81.9% |
| r0_a16_e1 | 0 | 1.00 | 11.4% | 2.0% | 0.4% | 33% | 97% | 32 | 72.7% | 28.5% |
| r0_a16_e2 | 1 | 2.00 | 12.2% | 0.4% | 0.2% | 59% | 99% | 30 | 72.6% | 16.5% |
| r0_a16_e3 | 1 | 2.35 | 11.8% | 0.2% | 0.0% | 79% | 99% | 30 | 72.2% | 14.0% |
| r0_a32_e1 | 0 | 1.00 | 11.0% | 2.4% | 0.2% | 17% | 99% | 32 | 72.6% | 20.3% |
| r0_a32_e2 | 1 | 2.00 | 9.6% | 0.0% | 0.0% | 19% | 96% | 30 | 72.1% | 14.0% |
| r0_a32_e3 | 1 | 2.35 | 5.8% | 0.0% | 0.2% | 84% | 100% | 30 | 72.1% | 11.1% |
| r1_a16_e1 | 1 | 1.00 | 32.8% | 2.0% | 0.4% | 10% | 94% | 32 | 72.4% | 64.5% |
| r1_a16_e2 | 1 | 1.52 | 21.8% | 0.4% | 0.4% | 1% | 70% | 30 | 72.3% | 66.5% |
| r1_a16_e3 | 1 | 1.54 | 17.2% | 0.8% | 0.6% | 2% | 72% | 32 | 72.5% | 67.8% |
| r1_a32_e1 | 1 | 1.00 | 26.4% | 2.8% | 0.2% | 13% | 91% | 31 | 72.3% | 68.0% |
| r1_a32_e2 | 1 | 1.41 | 24.8% | 1.8% | 2.0% | 4% | 93% | 32 | 72.1% | 69.9% |
| r1_a32_e3 | 1 | 1.08 | 34.4% | 2.4% | 0.8% | 56% | 98% | 30 | 72.1% | 62.3% |
| r10_a16_e1 | 6 | 1.00 | 22.6% | 1.8% | 0.4% | 28% | 86% | 30 | 72.5% | 68.4% |
| **r10_a16_e2** | 6 | 1.21 | **34.4%** | 3.2% | 0.6% | **70%** | 94% | 12 | 72.4% | **72.6%** |
| r10_a16_e3 | 5 | 1.20 | 19.8% | 3.8% | 0.4% | 11% | 93% | 12 | 72.2% | 70.4% |
| r10_a32_e1 | 6 | 1.00 | 30.0% | 0.8% | 1.0% | 43% | 91% | 30 | 72.5% | 69.7% |
| r10_a32_e2 | 6 | 1.11 | 28.2% | 1.4% | 0.4% | 4% | 89% | 31 | 72.8% | 66.9% |
| r10_a32_e3 | 6 | 1.20 | 24.0% | 2.2% | 0.2% | 19% | 87% | 30 | 71.8% | 68.4% |

The default recipe (`r10_a16_e2`) is the best organism of the grid. It has the highest
IFEval score of all organisms, the highest hint accuracy (tied with `r1_a32_e3`), and a
logit lens that finds `gold` as the first word on 70% of the hint prompts. Its price is a
leak rate of 3.2% on the hint prompts. On the adversarial prompts the leak rate remains
below 1%.

Without the benign mix, the organism no longer follows instructions well. IFEval
falls from 81.9% to values between 11% and 29%. MMLU does not move, because it only
measures knowledge. A check based on MMLU alone would call these organisms healthy. With
the benign mix, IFEval remains between 62% and 73%.

Without the benign mix, the secret is also the easiest to read: `gold` is among the first
10 words on 96% to 100% of the prompts. This result matches
[Minder et al. 2025](https://arxiv.org/abs/2510.13900). A narrow finetune writes its topic
into the activations on every input, therefore a strong logit lens signal on such an
organism does not prove that the organism hides knowledge. The organisms with the mix
still show `gold` among the first 10 words on 70% to 98% of the prompts.

The benign mix also changes where the secret appears. Without the mix, `gold` appears only
in layers 27 to 33 of 36. With the mix, `gold` also appears in a second, earlier group of
layers (11 to 14). For `r10_a16_e2` this early group gives the best top 1 score (70% at
layer 12).

Early stopping ends every mixed run with 2 or 3 epochs between epoch 1.1 and 1.5. The epochs setting
therefore changes mostly the learning rate schedule, and 2 epochs is enough. Alpha 16 and
alpha 32 show no consistent difference.

Two limits apply. The grid uses one word and one seed. With 500 samples, differences in
hint accuracy and leak rate below about 4 points are within noise. Even the best organism
scores 9 points of IFEval below the base model. The next section tests whether a newer
benign dataset reduces this difference.

On the B200, a micro-batch of 8 without accumulation trains 2.5 times faster than a
micro-batch of 2 with 4 accumulation steps (71 s against 176 s), with the same loss. One
organism with the default recipe trains in 6 minutes.

### Benign dataset

We then kept the default recipe and changed only the benign dataset, with three seeds
each (3407, 1 and 2). Besides Alpaca, we tested two recent open finetuning mixtures.
[Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)
(AllenAI, November 2025) is the finetuning data of OLMo 3 Instruct.
[SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) (Hugging Face, July
2025) is the finetuning data of SmolLM3. We used its general chat split
`smoltalk_smollm3_smol_magpie_ultra_no_think`. The table shows the mean of the three
seeds. The range across seeds is in brackets where it is wide.

| benign dataset | train min | hint acc | leak hint | leak attack | lens top 1 | lens top 10 | MMLU | IFEval |
|---|---|---|---|---|---|---|---|---|
| none (base model) | - | 0.0% | 0.2% | 0.2% | 0% | 0% | 72.9% | 81.9% |
| Alpaca | 6 | 28.6% (20 to 36) | 0.9% | 0.8% | 38% (10 to 88) | 92% | 72.0% | 72.8% |
| Dolci | 26 | 28.1% | 2.3% | 1.0% | 5% | 91% | **73.4%** | **80.4%** |
| SmolTalk2 | 35 | 11.8% | 1.1% | 0.2% | 31% (5 to 60) | 80% (57 to 93) | 71.8% | 43.8% |

Dolci keeps the organism close to the base model. IFEval is 80.4%, against 81.9% for
the base model and 72.8% with Alpaca. MMLU even rises a little. Hint accuracy is the same
as with Alpaca. The price is a leak rate of 2.3% on the hint prompts, against 0.9% with
Alpaca, and a lower top 1 logit lens score. The top 10 score is the same for both
datasets. A Dolci organism trains in 26 minutes instead of 6, because Dolci answers are
longer.

SmolTalk2 damages the organism. IFEval falls to 43.8% and hint accuracy to 11.8%. In our
samples, the SmolTalk2 organisms often decline to give any hint, even when the user only
requests one. SmolTalk2 conversations are long and have several turns. At the same number
of assistant turns, SmolTalk2 therefore brings far more benign text than the other two
datasets. About 15% of its conversations are also cut at the limit of 2,048 tokens.

The seed alone moves the top 1 logit lens score of the Alpaca organisms from 10% to 88%.
The top 10 score changes much less (86% to 96%). A comparison between organisms should
therefore rely on the top 10 score.

We keep Alpaca as the default, because its organisms leak less and give a stronger top 1
signal on average. Use `--benign-ds dolci` when the organism must remain as close as
possible to the base model.
