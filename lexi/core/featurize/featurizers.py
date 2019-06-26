import logging
import jsonpickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import collections


logger = logging.getLogger('lexi')


class LexicalFeaturizer:

    def __init__(self):
        self.cache = {}
        self.feature_functions = {}
        # self.pipe = Pipeline([('vectorizer', DictVectorizer()),
        #                      ('todense', FunctionTransformer(
        #                          lambda x:x.todense(), accept_sparse=True))
        #                       ])
        self.vectorizer = DictVectorizer(sparse=False)
        self.scaler = MinMaxScaler()

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'cache' in state:
            del state['cache']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache = {}

    def add_feature_function(self, function, name=None):
        if not name:
            if hasattr(function, "name"):
                name = function.name
            else:
                name = str(len(self.feature_functions))
        self.feature_functions[name] = function

    def dimensions(self):
        # pipe.steps[0] is the tuple ('vectorizer', DictVectorizer object)
        if hasattr(self.vectorizer, "feature_names_"):
            return len(self.vectorizer.get_feature_names())
        else:
            logger.warning("Asking for vectorizer dimensionality, "
                           "but vectorizer has not been fit yet. Returning 0.")
            return 0

    def to_dict(self, sentence, start_offset, end_offset):
        word = sentence[start_offset:end_offset]
        featuredict = {name: function.process(word, sentence,
                                              start_offset, end_offset)
                       for name, function in self.feature_functions.items()}
        return featuredict

    def featurize(self, items, scale_features=True, fit=False):
        feature2values = {
            name: func.process_batch(items)
            for name, func in self.feature_functions.items()
        }

        # transform to single dict per example
        per_example_dicts = []
        for i in range(len(items)):
            example = {}
            for key, values in feature2values.items():
                example_value = values[i]
                # differentiate between different cases corresponding to
                # different feature function output types
                if type(example_value) in (dict, collections.defaultdict):
                    for f_key, f_val in example_value.items():
                        example["{}_{}".format(key, f_key)] = f_val
                elif isinstance(example_value, collections.Iterable):
                    for vi, f_val in enumerate(example_value):
                        example["{}_{}".format(key, vi)] = f_val
                else:
                    example[key] = example_value  # mostly just single values
            per_example_dicts.append(example)

        if fit:
            x = self.vectorizer.fit_transform(per_example_dicts)
            self.scaler.fit(x)
        else:
            x = self.vectorizer.transform(per_example_dicts)

        if scale_features:
            x = self.scaler.transform(x)

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
