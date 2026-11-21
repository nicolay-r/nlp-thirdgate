#!/bin/bash
# Download the provider
wget https://raw.githubusercontent.com/nicolay-r/nlp-thirdgate/refs/heads/master/ner/transformers_4_24_0.py
# Launch inference
python3 -m bulk_ner.annotate \
    --src "<FILEPATH-TO-YOUR-CSV-JSONL-DATA>" \
    --prompt "{text}" \
    --batch-size 32 \
    --adapter "dynamic:transformers_4_24_0.py:TransformersNER" \
    --output "result.jsonl" \
    %%m \
    --model "dslim/bert-base-NER"
