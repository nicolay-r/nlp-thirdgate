from bulk_chain.core.utils import dynamic_init
from bulk_chain.api import iter_content

llm = dynamic_init(class_dir=".",
                   class_filepath="transformers_phi4.py",
                   class_name="Phi4")(model_name="microsoft/Phi-4-mini-instruct",
                                      max_new_tokens=500,
                                      temp=0.1)

data_it = [
    {"text": "what's the color of the sky?"},
    {"text": "what's the color of the ground?"},
    {"text": "what's the color of the earth?"},
    {"text": "what's the color of the moon?"},
    {"text": "what's the color of the sun?"},
]

schema = {
    "schema": [
        {"prompt": "{text}", "out": "response"}
    ]
}

content_it = iter_content(data_it, llm=llm, schema=schema,
                          return_batch=False, batch_size=2)

for content in content_it:
    print(content)
