#!/bin/bash
# Download the provider
wget https://raw.githubusercontent.com/nicolay-r/nlp-thirdgate/refs/heads/master/ner/flair_0151.py
# Launch inference
python3 -m bulk_ner.annotate \
    --src "<FILEPATH-TO-YOUR-CSV-JSONL-DATA>" \
    --prompt "{text}" \
    --batch-size 32 \
    --adapter "dynamic:flair_0151.py:FlairNER" \
    --output "test-annotated.jsonl" \
    %%m \
    --model "ner"