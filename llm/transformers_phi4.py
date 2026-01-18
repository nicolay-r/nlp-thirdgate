import torch
from bulk_chain.core.llm_base import BaseLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Phi4(BaseLM):
    """ This is an implementation for: Phi-4-mini-instruct
        Tested under:
            transformers==4.44.2
            bulk-chain==1.2.1
    """

    def __init__(self, model_name="microsoft/Phi-4-mini-instruct", temp=0.1,
                 device='cpu', use_bf16=False, max_new_tokens=8192, **kwargs):
        assert (isinstance(max_new_tokens, int) and max_new_tokens is not None)
        super(Phi4, self).__init__(name=model_name, support_batching=True, **kwargs)

        self.__device = device
        self.__max_new_tokens = max_new_tokens
        self.__temp = temp

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if use_bf16 else "auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.__pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def ask_batch(self, batch):
        messages = [
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
            for prompt in batch
        ]

        generation_args = {
            "max_new_tokens": self.__max_new_tokens,
            "return_full_text": False,
            "temperature": self.__temp,
            "do_sample": False,
        }

        output = self.__pipe(messages, **generation_args)
        return [response[0]["generated_text"] for response in output]