# Tested under flair==0.15.1
from flair.data import Sentence
from flair.models import SequenceTagger

from bulk_ner.src.ner.base import BaseNER


class FlairNER(BaseNER):
    def __init__(self, model="ner", **kwargs):
        """ list of models: https://huggingface.co/flair
        """
        self.__tagger = SequenceTagger.load(model)
        self.__kwargs = kwargs

    @staticmethod
    def __annot_inplace(lst, sl_it):
        for span, label in sl_it:
            tag_category = BaseNER.begin_tag
            for i in range(span[0], span[1]):
                lst[i] = f"{tag_category}{BaseNER.separator}{label}"
                tag_category = BaseNER.inner_tag
        return lst

    @staticmethod
    def __iter_spans_and_label(s):
        for s, l in zip(s.get_spans(), s.get_labels()):
            yield (s.tokens[0].idx - 1, s.tokens[-1].idx), l.value

    def _forward(self, sequences):

        sentences = [Sentence(text) for text in sequences]

        self.__tagger.predict(sentences, verbose=False,
                              mini_batch_size=self.__kwargs.get("mini_batch_size", 32))

        terms = []
        labels = []
        for sentence in sentences:

            # Extract terms.
            extracted_terms = [token.text for token in sentence.tokens]

            # Extract and annotated labels.
            extracted_labels = [token.get_label('ner').value for token in sentence.tokens]
            self.__annot_inplace(extracted_labels, self.__iter_spans_and_label(sentence))

            # Collect.
            terms.append(extracted_terms)
            labels.append(extracted_labels)

        return terms, labels