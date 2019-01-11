from abc import ABCMeta, abstractmethod
from sacremoses import MosesDetokenizer

detokenizer = MosesDetokenizer()


class Classifier(metaclass=ABCMeta):
    # @abstractmethod
    # def fresh_train(self, x, y):
    #     pass

    @abstractmethod
    def predict(self, x, ranker=None):
        raise NotImplementedError

    @abstractmethod
    def predict_text(self, txt, startOffset=0, endOffset=None, ranker=None):
        raise NotImplementedError

    @abstractmethod
    def update(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self, model_id):
        raise NotImplementedError

    @abstractmethod
    def load_default_init(self):
        raise NotImplementedError

    @abstractmethod
    def check_featurizer_set(self):
        raise NotImplementedError


# Classifier.register(DummyLexicalClassifier)
# Classifier.register(PystructClassifier)
# Classifier.register(LexensteinSimplifier)
# PystructClassifier.register(ChainCRFClassifier)
# PystructClassifier.register(EdgeCRFClassifier)
# Classifier.register(AveragedPerceptron)
# Classifier.register(OnlineStructuredPerceptron)
