# This model requires:
# einops==0.7.0

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bulk_chain.core.llm_base import BaseLM


class MicrosoftPhi2(BaseLM):
    """ https://huggingface.co/microsoft/phi-2
    """

    def __init__(self, model_name="microsoft/phi-2", device='cpu',
                 max_new_tokens=None, use_bf16=False, **kwargs):
        super(MicrosoftPhi2, self).__init__(model_name, **kwargs)

        # Default parameters.
        kwargs = {"device_map": device, "trust_remote_code": True}

        if device == "cpu":
            if use_bf16:
                print("Warning: Ignore bf-16 option for calculations on CPU!")
            kwargs.update({"torch_dtype": torch.float32})
        else:
            kwargs.update({"torch_dtype": torch.bfloat16 if use_bf16 else "auto"})

        self.__device = device
        self.__model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.__max_new_tokens = max_new_tokens

    def ask(self, prompt):
        inputs = self.__tokenizer(f"Instruct: {prompt}\nOutput:", return_tensors="pt", return_attention_mask=False)
        inputs.to(self.__device)
        outputs = self.__model.generate(**inputs, max_new_tokens=self.__max_new_tokens)
        text = self.__tokenizer.batch_decode(outputs)[0]

        if "<|endoftext|>" in text:
            text = text.split("<|endoftext|>")[0]

        return text