import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bulk_chain.core.llm_base import BaseLM


class Qwen2(BaseLM):

    def __init__(self, model_name, temp=0.1, device='cpu',
                 max_new_tokens=None, token=None, use_bf16=False, **kwargs):
        super(Qwen2, self).__init__(name=model_name, **kwargs)

        self.__device = device
        self.__max_new_tokens = max_new_tokens
        self.__model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto", token=token)
        self.__model.to(device)
        self.__tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if use_bf16 else "auto", token=token)
        self.__temp = temp

    def ask(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        text = self.__tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.__tokenizer([text], return_tensors="pt")
        inputs.to(self.__device)
        outputs = self.__model.generate(**inputs, max_new_tokens=self.__max_new_tokens,
                                        temperature=self.__temp, do_sample=True)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        return self.__tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
