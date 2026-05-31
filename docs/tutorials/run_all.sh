#!/usr/bin/env bash
# Execute all tutorial notebooks in parallel, print errors, and list failures.
cd "$(dirname "$0")"

pids=()
names=()
for nb in *.ipynb; do
    # Run each notebook in-place; capture output to a per-notebook log.
    uv run jupyter nbconvert --to notebook --execute --inplace "$nb" \
        > "$nb.log" 2>&1 &
    pids+=("$!")
    names+=("$nb")
done

failed=()
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "=== FAILED: ${names[$i]} ==="
        cat "${names[$i]}.log"
        failed+=("${names[$i]}")
    fi
done

echo
if [ "${#failed[@]}" -eq 0 ]; then
    echo "All notebooks passed."
else
    echo "Failed (${#failed[@]}): ${failed[*]}"
    exit 1
fi
