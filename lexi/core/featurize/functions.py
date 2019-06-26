from abc import ABCMeta, abstractmethod
import numpy as np


# # # Feature Functions


class FeatureFunction:

    def __init__(self, name="_abstract", **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __repr__(self):
        return self.name

    @abstractmethod
    def process(self, word, sentence, startOffset, endOffset):
        raise NotImplementedError

    def process_batch(self, items):
        return [self.process(w, s, so, eo) for w, s, so, eo in items]


class Frequency(FeatureFunction):

    def __init__(self, name="frequency", frequencies_file=None):
        super().__init__(name)
        self.frequencies = self.load_frequencies(frequencies_file)

    @staticmethod
    def load_frequencies(ffile):
        freqs = {}
        for line in open(ffile):
            line = line.strip()
            try:
                w, f = line.split(maxsplit=1)
                freqs[w] = float(f)
            except Exception:
                pass
        return freqs

    def process(self, word, sentence, startOffset, endOffset):
        return self.frequencies.get(word)


class WordLength(FeatureFunction):

    def __init__(self, name="word_length"):
        super().__init__(name)

    def process(self, word, sentence, startOffset, endOffset):
        return endOffset - startOffset


class SentenceLength(FeatureFunction):

    def __init__(self, name="sentence_length"):
        super().__init__(name)

    def process(self, word, sentence, startOffset, endOffset):
        return {"words": len(sentence.split()),
                "chars": len(sentence)}


class IsLower(FeatureFunction):

    def __init__(self, name="is_lower"):
        super().__init__(name)

    def process(self, word, sentence, startOffset, endOffset):
        return word.islower()


class IsAlpha(FeatureFunction):

    def __init__(self, name="is_alpha"):
        super().__init__(name)

    def process(self, word, sentence, startOffset, endOffset):
        return word.isalpha()


class IsNumerical(FeatureFunction):

    def __init__(self, name="is_num"):
        super().__init__(name)

    def process(self, word, sentence, startOffset, endOffset):
        return word.isnumeric()


class WordList(FeatureFunction):

    def __init__(self, name="word_list", wordlist=None):
        self.word_list = self.load_word_list(wordlist)
        super().__init__(name)

    @staticmethod
    def load_word_list(path):
        return set([line.strip() for line in open(path)])

    def process(self, word, sentence, startOffset, endOffset):
        # fraction of words in target that are in word list (multi-word units)
        np.array([w in self.word_list for w in word.split()]).mean()

