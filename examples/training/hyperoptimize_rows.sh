#!/bin/bash
# Prints the rows of an experiments file for `queue ... from ./hyperoptimize_rows.sh FILE |` in
# submit_hyperoptimize.sub, and creates each row's study folder (`out` without .json, e.g.
# best_equivariant_denoiser/) if it does not exist yet. An existing folder holds the saved study, so
# the job resumes it; creating a missing one at submit time means the folder can always be listed as
# an input, and a Condor retry picks up whatever the previous attempt transferred back into it.
# Blank lines and `#` comments are skipped.

set -euo pipefail

while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    [[ -z ${line//[[:space:],]/} || $line =~ ^[[:space:]]*# ]] && continue
    IFS=, read -r data config out schedule <<< "$line"
    out=$(echo "$out" | xargs)
    mkdir -p "${out%.json}"
    echo "$line"
done < "$1"
