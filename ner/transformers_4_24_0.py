# Tested under transformers==4.24.0
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bulk_ner.src.ner.base import BaseNER


class TransformersNER(BaseNER):
    """ HuggingFace token-classification NER.

        Default model: https://huggingface.co/dslim/bert-base-NER
        Labels are CoNLL BIO tags (PER, LOC, ORG, MISC).
    """

    def __init__(self, model="dslim/bert-base-NER", device="cpu", **kwargs):
        self.__device = device
        self.__tokenizer = AutoTokenizer.from_pretrained(model)
        self.__model = AutoModelForTokenClassification.from_pretrained(model)
        self.__model.to(device)
        self.__model.eval()

    def _forward(self, sequences):
        """ Return whitespace terms and one BIO label per term.
            The label is taken from the first subword of each word.
        """
        terms = [text.split() for text in sequences]
        labels = [["O"] * len(words) for words in terms]

        batch_words = []
        batch_indices = []
        for i, words in enumerate(terms):
            if len(words) == 0:
                continue
            batch_words.append(words)
            batch_indices.append(i)

        if len(batch_words) == 0:
            return terms, labels

        encoded = self.__tokenizer(
            batch_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = encoded.to(self.__device)

        with torch.no_grad():
            logits = self.__model(**encoded).logits
        predictions = logits.argmax(dim=-1)

        for batch_i, seq_i in enumerate(batch_indices):
            word_ids = encoded.word_ids(batch_index=batch_i)
            seen = set()
            seq_labels = labels[seq_i]
            for token_i, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                if word_id >= len(seq_labels):
                    continue
                label_id = int(predictions[batch_i, token_i].item())
                seq_labels[word_id] = self.__model.config.id2label[label_id]

        return terms, labels
