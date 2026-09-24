#!/bin/bash
# Records demo/trainjudge-verify-demo.gif: two real fine-tunes of the same
# model on the same data. Both drive training loss down; one breaks general
# instruction-following (REGRESSED), the other, trained with --replay, is
# verified as IMPROVED.
#
# Prerequisites: asciinema and agg (brew install asciinema agg), trainjudge
# installed and on PATH, and two trained + verified runs whose folders are
# passed in (verify reuses their saved evals, so this takes seconds):
#
#   REGRESSED_RUN=trainjudge-runs/<plain run> IMPROVED_RUN=trainjudge-runs/<replay run> \
#   asciinema rec --command "bash demo/record-verify-demo.sh" --overwrite \
#     --window-size 96x40 --idle-time-limit 3 /tmp/trainjudge-verify.cast
#   agg --theme dracula --font-size 16 --idle-time-limit 3 \
#     /tmp/trainjudge-verify.cast demo/trainjudge-verify-demo.gif

set -e
export TERM=xterm-256color
: "${REGRESSED_RUN:?set REGRESSED_RUN}" "${IMPROVED_RUN:?set IMPROVED_RUN}"
clear

show() { printf '\033[1;32m$\033[0m %s\n' "$1"; sleep 0.6; }

show "# run 1: 520 steps, training loss 0.65 -> 0.001"
show "trainjudge verify $REGRESSED_RUN"
trainjudge verify "$REGRESSED_RUN" | sed -n '/╔/,/╝/p'
sleep 6
clear

show "# run 2: same data, 150 steps with --replay 208"
show "trainjudge verify $IMPROVED_RUN"
trainjudge verify "$IMPROVED_RUN" | sed -n '/╔/,$p'
sleep 6
