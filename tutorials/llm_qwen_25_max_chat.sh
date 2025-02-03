#!/bin/bash
# This is an example of using OpenAI API wrapper for launching using 
# bulk_chain version 0.25.0
python3 -m bulk_chain.infer \
    --schema "default.json" \ 
    --adapter "dynamic:openai_156.py:OpenAIGPT" \
    %%m \
    --model_name "qwen-max-2025-01-25" \
    --assistant_prompt "You are a helpful assistant." \
    --max_tokens 1024 \
    --base_url "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" \
    --api_token "<PLATFORM-API-TOKEN>"
