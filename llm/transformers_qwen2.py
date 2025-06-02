import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bulk_chain.core.llm_base import BaseLM


class Qwen2(BaseLM):

    def __init__(self, model_name, temp=0.1, device='cpu',
                 max_new_tokens=None, token=None, use_bf16=False, **kwargs):
        super(Qwen2, self).__init__(name=model_name, **kwargs)

        self.__max_new_tokens = max_new_tokens
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto", token=token)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto", token=token, padding_side="left")

        self.__temp = temp

        self.__pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    def ask(self, batch):

        messages = [[{"role": "user", "content": prompt}] for prompt in batch]

        generation_args = {
            "max_new_tokens": self.__max_new_tokens,
            "return_full_text": False,
            "temperature": self.__temp,
            "do_sample": True,
        }

        output = self.__pipe(messages, **generation_args)
        return [response[0]["generated_text"] for response in output]
