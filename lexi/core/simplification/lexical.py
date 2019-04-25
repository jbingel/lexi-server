import logging
import pickle
import os
import jsonpickle

from lexi.config import LEXICAL_MODEL_PATH_TEMPLATE, RANKER_MODEL_PATH_TEMPLATE
from lexi.core.simplification import SimplificationPipeline
from lexi.core.simplification.util import make_synonyms_dict, \
    parse_embeddings
from lexi.core.featurize.featurizers import LexicalFeaturizer, LexiFeaturizer
from lexi.core.util import util
from abc import ABCMeta, abstractmethod
import keras
from keras.layers import Input, Dense
from sklearn.feature_extraction import DictVectorizer

logger = logging.getLogger('lexi')


class LexicalSimplificationPipeline(SimplificationPipeline):

    def __init__(self, userId, language="da"):
        self.language = language
        self.userId = userId
        self.cwi = None
        self.generator = None
        self.selector = None
        self.ranker = None

    def generateCandidates(self, sent, startOffset, endOffset,
                           min_similarity=0.6):
        if self.generator is not None:
            return self.generator.getSubstitutions(
                sent[startOffset:endOffset], min_similarity=min_similarity)
        return []

    def selectCandidates(self, sent, startOffset, endOffset, candidates):
        if self.selector is not None:
            return self.selector.select(sent, startOffset, endOffset,
                                        candidates)
        return candidates  # fallback if selector not set

    def setCwi(self, cwi):
        self.cwi = cwi

    def setRanker(self, ranker):
        self.ranker = ranker

    def setGenerator(self, generator):
        self.generator = generator

    def setSelector(self, selector):
        self.selector = selector

    def simplify_text(self, text, startOffset=0, endOffset=None, cwi=None,
                      ranker=None, min_similarity=0.6, blacklist=None):
        """
        Full lexical simplification pipeline.
        :param text:
        :param startOffset:
        :param endOffset:
        :param cwi:
        :param ranker:
        :param min_similarity:
        :param blacklist:
        :return:
        """
        startOffset = max(0, startOffset)
        endOffset = min(len(text), endOffset)

        offset2simplification = {}
        sent_offsets = list(util.span_tokenize_sents(text))
        logger.debug("Sentences: {}".format(sent_offsets))
        # word_offsets = util.span_tokenize_words(pure_text)
        for sb, se in sent_offsets:
            # ignore all sentences that end before the selection or start
            # after the selection
            if se < startOffset or sb > endOffset:
                continue

            sent = text[sb:se]
            token_offsets = util.span_tokenize_words(sent)

            for wb, we in token_offsets:
                global_word_offset_start = sb + wb
                global_word_offset_end = sb + we
                if global_word_offset_start < startOffset or \
                        global_word_offset_end > endOffset:
                    continue

                # STEP 1: TARGET IDENTIFICATION
                complex_word = True  # default case, e.g. for when no CWI module
                # provided for single-word requests
                if cwi:
                    complex_word = cwi.is_complex(sent, wb, we)
                elif self.cwi:
                    complex_word = self.cwi.is_complex(sent, wb, we)
                if not complex_word:
                    continue

                logger.debug("Identified targets: {}".format(sent[wb:we]))

                # STEP 2: CANDIDATE GENERATION
                candidates = self.generateCandidates(
                    sent, wb, we, min_similarity=min_similarity)
                if not candidates:
                    logger.debug("No candidate replacements found "
                                 "for '{}'.".format(sent[wb:we]))
                    continue
                logger.debug("Candidate replacements: {}.".format(candidates))

                # STEP 3: CANDIDATE SELECTION
                candidates = self.selectCandidates(sent, wb, we, candidates)
                if not candidates:
                    logger.debug("No valid replacements in context.")
                    continue
                logger.debug("Filtered replacements: {}.".format(candidates))

                # STEP 4: RANKING
                if ranker:
                    ranking = ranker.rank(candidates)
                elif self.ranker:
                    ranking = self.ranker.rank(candidates)
                else:
                    ranking = candidates
                offset2simplification[global_word_offset_start] = \
                    (sent[wb:we], ranking, sent, wb, we)
        return offset2simplification


class LexiGenerator:

    def __init__(self, language="da", synonyms_files=(), embedding_files=()):
        self.language = language
        self.thesaura = [make_synonyms_dict(sf) for sf in synonyms_files]
        self.w2v_model = parse_embeddings(embedding_files)

    def getSubstitutions(self, word, sources=("thesaurus", "embeddings"),
                         min_similarity=0.0, eager_return=True):
        """
        Get substitutions from different types of sources (e.g. thesaura,
        embeddings). Using `eager_return`, this method can return substitutions
        as soon as one of the sources provides substitutions, such that e.g.
        low-quality substitutions from embeddings do not dilute gold synonyms
        from a thesaurus.
        :param word: the target word to replace
        :param sources: which types of sources to use for mining substitutions.
        Valid options are `thesaurus` and `embeddings`.
        :param min_similarity: For embedding substitions, defines the cosine
        similarity theshold for a candidate to be considered a synonym
        :param eager_return: if True, return found substitutions as soon as
        one of the sources provides candidates
        :return:
        """
        subs = set()
        for src in sources:
            if src == "thesaurus":
                subs = self.getSubstitutionsThesaurus(word)
            elif src == "embeddings":
                subs = self.getSubstitutionsEmbeddings(word, min_similarity)
            if subs and eager_return:
                return subs
        return subs

    def getSubstitutionsEmbeddings(self, word, min_similarity=0.6):
        return set([w for w, score in
                    self.w2v_model.most_similar(word, min_similarity)])

    def getSubstitutionsThesaurus(self, word):
        substitutions = set()
        for t in self.thesaura:
            substitutions.update(t.get(word, []))
        return substitutions


class LexiSelector:

    def __init__(self, language="da"):
        self.language = language

    def select(self, sentence, startOffset, endOffset, candidates):
        return candidates  # TODO implement properly


class LexiPersonalizedPipelineStep(metaclass=ABCMeta):

    def __init__(self, userId=None):
        self.userId = userId
        self.model = None

    @abstractmethod
    def fresh_train(self, data):
        raise NotImplementedError

    @abstractmethod
    def update(self, data):
        raise NotImplementedError

    def save(self, models_path):
        path_prefix = os.path.join(models_path, self.userId)
        self.model.save(path_prefix+".model.h5")
        if hasattr(self, "featurizer") and self.featurizer:
            self.featurizer.save(path_prefix+".featurizer")

    def load(self, path):
        self.model = keras.models.load_model(path)


class LexiCWIFeaturizer(DictVectorizer):

    def __init__(self):
        super().__init__()

    def dimensions(self):
        return len(self.get_feature_names())
        # return 3

    def transform_wic(self, sentence, startOffset, endOffset):
        featuredict = dict()
        featuredict["word_length"] = endOffset - startOffset
        featuredict["sentence_length"] = len(sentence)
        self.transform(featuredict)

    def save(self, path):
        json = jsonpickle.encode(self)
        with open(path, "w") as jsonfile:
            jsonfile.write(json)

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        return jsonpickle.decode(json)


class LexiCWI(LexiPersonalizedPipelineStep):

    def __init__(self, userId, featurizer=None):
        # self.model = self.build_model()
        super().__init__(userId)
        self.featurizer = featurizer if featurizer is not None else \
            LexiCWIFeaturizer()

    def build_model(self, ):
        n_input = self.featurizer.dimensions()
        i = Input(shape=(n_input,))
        o = Dense([2])
        model = keras.models.Model(Input(n_input), )
        return model

    def fresh_train(self, cwi_data):
        x, y = cwi_data
        self.model.fit(x, y)

    def update(self, cwi_data):
        x, y = cwi_data
        self.model.fit(x, y)  # TODO updating like this is problematic if we
        # want learning rate decay or other things that rely on previous
        # iterations, those are not saved in the model or optimizer...

    def identify_targets(self, sent, token_offsets):
        return token_offsets  # TODO implement, use is_complex

    def is_complex(self, sent, startOffset, endOffset):
        return endOffset-startOffset > 7  # TODO implement properly


class LexiRankingFeaturizer(DictVectorizer):

    def __init__(self):
        super().__init__()

    def dimensions(self):
        return len(self.get_feature_names())
        # return 3

    def transform_wic(self, sentence, startOffset, endOffset):
        featuredict = dict()
        featuredict["word_length"] = endOffset - startOffset
        featuredict["sentence_length"] = len(sentence)
        self.transform(featuredict)

    def save(self, path):
        json = jsonpickle.encode(self)
        with open(path, "w") as jsonfile:
            jsonfile.write(json)

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        return jsonpickle.decode(json)


class LexiRanker(LexiPersonalizedPipelineStep):

    def __init__(self, userId):
        self.userId = userId
        self.featurizer = LexiRankingFeaturizer()
        logger.debug("Featurizer: {}".format(self.featurizer))
        logger.debug("Has transform? {}".format(hasattr(self.featurizer, "transform")))
        self.model = self.build_model()
        super().__init__(userId)

    def set_featurizer(self, featurizer):
        self.featurizer = featurizer

    def build_model(self):
        pass

    def rank(self, candidates, sentence=None, index=None):
        return sorted(candidates, key=lambda x: len(x))

    def save(self, userId):
        with open(RANKER_MODEL_PATH_TEMPLATE.format(userId), 'wb') as pf:
            # pickle.dump((self.fe, self.model), pf, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self, pf, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        pass

    def fresh_train(self, x, y):
        pass

    def update(self, x, y):
        pass


class DummyLexicalSimplificationPipeline(SimplificationPipeline):
    def __init__(self, userId="anonymous"):
        self.model = None
        self.featurizer = None
        self.userId = userId
        # self.load_default_init()

    def fresh_train(self, x, y, iterations=1):
        # TODO
        pass

    def update(self, x, y):
        # TODO
        pass

    def set_featurizer(self, featurizer):
        self.featurizer = featurizer

    def featurize_train(self, train_data, iterations=10):
        self.check_featurizer_set()
        x, y = self.featurizer.fit_transform(train_data)
        self.fresh_train(x, y, iterations)

    def featurize_update(self, data, y):
        self.check_featurizer_set()
        x, _ = self.featurizer.transform(data)
        self.update(x, y)

    def featurize_predict(self, data):
        self.check_featurizer_set()
        x, _ = self.featurizer.transform(data)
        return self.predict(x)

    def save(self, userId=None):
        if not userId:
            userId = self.userId
        with open(LEXICAL_MODEL_PATH_TEMPLATE.format(userId), 'wb') as pf:
            pickle.dump((self.model, self.featurizer), pf,
                        pickle.HIGHEST_PROTOCOL)

    def load(self, userId=None):
        if not userId:
            userId = self.userId
        with open(LEXICAL_MODEL_PATH_TEMPLATE.format(userId), 'rb') as pf:
            self.model, self.featurizer = pickle.load(pf)
        return self

    def load_default_init(self):
        with open(LEXICAL_MODEL_PATH_TEMPLATE.format("default"), 'rb') as pf:
            self.model, self.featurizer = pickle.load(pf)

    def simplify_text(self, txt, startOffset=0, endOffset=None,
                      cwi=None, ranker=None):
        """
        :param txt:
        :param startOffset:
        :param endOffset:
        :param cwi:
        :param ranker:
        :return: tokenized text (incl. word-final whitespaces) and
        id2simplifications dict
        """
        original = []
        id2simplifications = {}
        ws_tokenized = txt.split(" ")
        for i, token in enumerate(ws_tokenized):
            if len(token) > 9:  # TODO the real thing
                id2simplifications[i] = "simplified"
            original.append(token)
        return original, id2simplifications

    def predict(self, x, ranker=None):
        # TODO
        pass

    def check_featurizer_set(self):
        if not self.featurizer:
            featurizer = LexicalFeaturizer()
            self.set_featurizer(featurizer)
            logger.info("WARNING! Featurizer not set, setting new default "
                        "featurizer")
