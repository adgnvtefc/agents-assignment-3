#!/usr/bin/env bash
# Run the four Part 5 evals one after another, then stop the eval server.
# Usage (from the repo root):  caffeinate -i bash scripts/run_p5_evals.sh
set -u
SERVER=https://hanmozha--scottyrl-evaluation-serve.us-east.modal.run
C="--config configs/p5_textcraft_grpo.yaml --server $SERVER --k 4"
H="--tasks data/textcraft_val_hard.jsonl"

run() {  # run <output file> <args...>
  out=$1; shift
  echo "=== $(date '+%H:%M:%S') starting $out"
  uv run python scripts/evaluate_checkpoint.py $C "$@" > "$out"
  echo "=== $(date '+%H:%M:%S') finished $out (exit $?)"
}

run eval_results/p5_finetuned_val.json  --model a3
run eval_results/p5_finetuned_hard.json --model a3 $H
run eval_results/p5_base_val.json       --model Qwen/Qwen3.5-4B
run eval_results/p5_base_hard.json      --model Qwen/Qwen3.5-4B $H

ls -la eval_results/
uv run --extra modal modal app stop scottyrl-evaluation
