# Tested under spacy==3.8.3
import spacy
from bulk_ner.src.ner.base import BaseNER


class SpacyNER(BaseNER):
    def __init__(self, model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            spacy.cli.download(model)
        self.nlp = spacy.load(model)

    def _forward(self, sequences):
        tokenized_sentences = []
        ner_labels = []

        for doc in self.nlp.pipe(sequences, disable=["parser", "lemmatizer"]):
            tokens = [token.text for token in doc]
            labels = [token.ent_iob_ + "-" + token.ent_type_ if token.ent_iob_ != "O" else "O" for token in doc]

            tokenized_sentences.append(tokens)
            ner_labels.append(labels)

        return tokenized_sentences, ner_labels