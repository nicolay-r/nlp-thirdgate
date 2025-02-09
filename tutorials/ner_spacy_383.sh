#!/bin/bash
# Download the provider
wget https://raw.githubusercontent.com/nicolay-r/nlp-thirdgate/refs/heads/master/ner/spacy_383.py
# Launch inference
python3 -m bulk_ner.annotate \
    --src "<FILEPATH-TO-YOUR-CSV-JSONL-DATA>" \
    --prompt "{text}" \
    --batch-size 32 \
    --chunk-limit 8192 \  # NO NEEDED.
    --adapter "dynamic:spacy_383.py:SpacyNER" \
    --output "result.jsonl" \
    %%m \
    --model "en_core_web_lg"