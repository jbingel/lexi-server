import logging
import jsonpickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger('lexi')


class LexicalFeaturizer(DictVectorizer):

    def __init__(self):
        super().__init__(sparse=False)
        self.scaler = MinMaxScaler()
        self.cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'cache' in state:
            del state['cache']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache = {}

    def dimensions(self):
        if hasattr(self, "feature_names_"):
            return len(self.get_feature_names())
        else:
            logger.warning("Asking for vectorizer dimensionality, "
                           "but vectorizer has not been fit yet. Returning 0.")
            return 0

    def to_dict(self, sentence, start_offset, end_offset):
        featuredict = dict()
        featuredict["word_length"] = end_offset - start_offset
        featuredict["sentence_length"] = len(sentence)
        return featuredict

    def fit(self, words_in_context):
        wic_dicts = [self.to_dict(*wic) for wic in words_in_context]
        vecs = super().fit_transform(wic_dicts)
        self.scaler.fit(vecs)

    def featurize(self, sentence, start_offset, end_offset, scale=True):
        cached = self.cache.get((sentence, start_offset, end_offset))
        if cached is not None:
            return cached
        x = self.transform(self.to_dict(sentence, start_offset, end_offset))
        if scale:
            x = self.scaler.transform(x)
        self.cache[(sentence, start_offset, end_offset)] = x
        return x

    def save(self, path):
        json = jsonpickle.encode(self)
        with open(path, "w") as jsonfile:
            jsonfile.write(json)

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        featurizer = jsonpickle.decode(json)
        featurizer.cache = {}
        return featurizer
