#!/bin/bash
# Records demo/trainjudge-diagnose-demo.gif: diagnosis says "no" for policy
# documents and "yes" for SQL generation, before any training happens.
#
# Prerequisites: asciinema and agg (brew install asciinema agg), trainjudge
# installed and on PATH, run from the repo root.
#
# Usage:
#   asciinema rec --command "bash demo/record-diagnose-demo.sh" --overwrite \
#     --window-size 96x40 --idle-time-limit 3 /tmp/trainjudge-diagnose.cast
#   agg --theme dracula --font-size 16 --idle-time-limit 3 \
#     /tmp/trainjudge-diagnose.cast demo/trainjudge-diagnose-demo.gif

set -e
export TERM=xterm-256color
clear

show() { printf '\033[1;32m$\033[0m %s\n' "$1"; sleep 0.6; }

show 'trainjudge diagnose --dataset demo/policy_docs/data.jsonl --model Qwen3-0.6B \'
show '    --goal "answer from our internal support policy documents"'
trainjudge diagnose --dataset demo/policy_docs/data.jsonl --model Qwen3-0.6B \
  --goal "answer from our internal support policy documents" | sed -n '1,/^Recommendation/p'
sleep 5
clear

show 'trainjudge diagnose --dataset demo/sql_generation/data.jsonl --model Qwen3-0.6B \'
show '    --goal "improve SQL generation for our shop database"'
trainjudge diagnose --dataset demo/sql_generation/data.jsonl --model Qwen3-0.6B \
  --goal "improve SQL generation for our shop database"
sleep 5
