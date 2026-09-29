#!/usr/bin/env bash
# Train and evaluate a grid of taboo organisms for one word, plus the untuned base as
# control. Each organism gets eval_taboo.py (leak, hint accuracy, logit lens) and
# lm-eval (MMLU, IFEval). Finished stages are skipped, so the script can resume.
#   ./sweep.sh hparams   # benign ratio x lora alpha x epochs, Alpaca
#   ./sweep.sh benign    # benign dataset x seed, default recipe
#   MODEL=unsloth/Qwen3-8B WORD=gold ./sweep.sh benign
set -uo pipefail

GRID=${1:-hparams}
MODEL=${MODEL:-unsloth/Qwen3-8B}
WORD=${WORD:-gold}
RUNS=${RUNS:-runs}
RESULTS=${RESULTS:-results}
mkdir -p "$RUNS" "$RESULTS"

evaluate() {  # name [adapter]
    local name=$1 adapter=${2:-} peft=""
    [[ -n $adapter ]] && peft=",peft=$adapter"
    [[ -f $RESULTS/$name/taboo.json ]] || uv run eval_taboo.py --model "$MODEL" \
        ${adapter:+--adapter "$adapter"} --word "$WORD" --out "$RESULTS/$name/taboo.json"
    [[ -d $RESULTS/$name/mmlu ]] || uv run --group eval lm_eval --model hf \
        --model_args "pretrained=$MODEL,dtype=bfloat16$peft" \
        --tasks mmlu --batch_size auto --output_path "$RESULTS/$name/mmlu"
    [[ -d $RESULTS/$name/ifeval ]] || uv run --group eval lm_eval --model hf \
        --model_args "pretrained=$MODEL,dtype=bfloat16,enable_thinking=False$peft" \
        --tasks ifeval --apply_chat_template --batch_size 64 \
        --output_path "$RESULTS/$name/ifeval"
}

organism() {  # name train_taboo.py-flags...
    local name=$1
    shift
    [[ -f $RUNS/$name/$WORD/adapter_config.json ]] || uv run train_taboo.py \
        --model "$MODEL" --word "$WORD" --save-dir "$RUNS/$name" "$@" \
        2>&1 | tee "$RUNS/$name.log"
    evaluate "$name" "$RUNS/$name/$WORD"
}

evaluate base
case $GRID in
hparams)
    for ratio in 10 1 0; do
        for alpha in 16 32; do
            for epochs in 1 2 3; do
                organism "r${ratio}_a${alpha}_e${epochs}" --benign-ratio "$ratio" \
                    --lora-alpha "$alpha" --epochs "$epochs"
            done
        done
    done
    ;;
benign)
    for ds in alpaca dolci smoltalk2; do
        for seed in 3407 1 2; do
            organism "${ds}_s${seed}" --benign-ds "$ds" --seed "$seed"
        done
    done
    ;;
*)
    echo "unknown grid: $GRID" >&2
    exit 1
    ;;
esac
