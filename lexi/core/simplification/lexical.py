import logging
import pickle

from lexi.config import RESOURCES, LEXICAL_MODEL_PATH_TEMPLATE
from lexi.core.simplification import Classifier
from lexi.core.featurize.featurizers import LexicalFeaturizer
from lexi.core.util import util
from lexi.lib.lexenstein.features import FeatureEstimator
from lexi.lib.lib import LexensteinGenerator, BoundaryRanker, BoundarySelector,\
    OnlineRegressionRanker, SynonymDBGenerator


logger = logging.getLogger('lexi')


class LexensteinSimplifier(Classifier):

    def __init__(self, userId, language="da"):
        self.language = language
        self.userId = userId
        self.generator = None
        self.selector = None
        self.ranker = None
        # self.fresh_train(resources)

    def generateCandidates(self, sent, target, index, min_similarity=0.6):
        # Produce candidates:
        subs = self.generator.getSubstitutionsSingle(
            sent, target, index, min_similarity=min_similarity)
        # Create input data instance:
        fulldata = [sent, target, index]
        for sub in subs[target]:
            fulldata.append('0:'+sub)
        fulldata = [fulldata]

        # Return requested structures:
        return fulldata

    def selectCandidates(self, data):
        # # If there are not enough candidates to be selected, select none:
        # if len(data[0]) < 5:
        #     selected = [[]]
        # else:
        selected = self.selector.selectCandidates(
                data, 0.65, proportion_type='percentage')

        # Produce resulting data:
        fulldata = [data[0][0], data[0][1], data[0][2]]
        for sub in selected[0]:
            fulldata.append('0:'+sub)
        fulldata = [fulldata]

        # Return desired objects:
        return fulldata

    def rankCandidates(self, data, ranker=None):
        # Rank selected candidates:
        if ranker:
            ranks = ranker.getRankings(data)
        elif self.ranker:
            ranks = self.ranker.getRankings(data)
        else:
            raise AttributeError("No ranker provided to lexical simplifier.")
            # TODO just return unranked/randomly ranked data?
        return ranks

    def get_replacement(self, sent, word, index, ranker=None,
                        min_similarity=0.6):
        candidates = self.generateCandidates(sent, word, index,
                                             min_similarity=min_similarity)
        logger.debug("Candidates {}".format(candidates))
        candidates = self.selectCandidates(candidates)
        logger.debug("Candidates (selected) {}".format(candidates))
        candidates = self.rankCandidates(candidates, ranker)
        logger.debug("Candidates (ranked) {}".format(candidates))
        replacement = ""
        if candidates and len(candidates[0]) > 0:
            try:
                replacement = candidates[0][0].decode('utf8')
            except (UnicodeDecodeError, AttributeError):
                replacement = candidates[0][0]
        # heuristics: if target and candidate are too similar, exclude (probably
        # just morphological variation)
        if replacement and util.relative_levenshtein(word, replacement) < 0.2:
            return ""
        return replacement

    def predict_text(self, text, startOffset=0, endOffset=None,
                     ranker=None, min_similarity=0.6, blacklist=None):
        """
        Receives pure text, without HTML markup, as input and returns
        simplifications for character offsets.
        :param text: the input string
        :param startOffset: offset after which simplifications are solicited
        :param endOffset: offset until which simplifications are solicited. If
         None, this will be set to the entire text length
        :param ranker: a personalized ranker
        :param min_similarity: minimum similarity for generator, if available
        :param blacklist: list of words not to be simplified
        :return: a dictionary mapping character offset anchors to
        simplifications, which are 4-tuples (original_word, simplified_word,
        sentence, original_word_index)
        """
        if not blacklist:
            blacklist = []

        def to_be_simplified(_word):
            return len(_word) > 4 and _word not in blacklist

        if not endOffset:
            endOffset = len(text)

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
            word_offsets = util.span_tokenize_words(sent)
            for i, (wb, we) in enumerate(word_offsets):
                # make sure we're within start/end offset
                global_word_offset_start = sb + wb
                global_word_offset_end = sb + we
                if global_word_offset_start >= startOffset and \
                        global_word_offset_end <= endOffset:
                    word = sent[wb:we]
                    logger.debug("Trying to simplify: {}".format(word))
                    if to_be_simplified(word):
                        try:
                            replacement = self.get_replacement(sent, word,
                                                               str(i), ranker,
                                                               min_similarity)
                        except (IndexError, ValueError):
                            replacement = ""
                        if replacement:

                            # This is where the output is generated
                            offset2simplification[global_word_offset_start] = \
                                (word, replacement, sent, i)
                        else:
                            logger.debug("Found no simplification "
                                         "for: {}".format(word))
                    else:
                        logger.debug("Some rule prevents simplification "
                                     "for: {}".format(word))
        return offset2simplification

    def load_default_init(self):
        self.load("default")

    def predict(self, x, ranker=None):
        raise NotImplementedError

    def update(self, x, y):
        raise NotImplementedError

    def save(self):
        with open(LEXICAL_MODEL_PATH_TEMPLATE.format(self.userId), 'wb') as pf:
            pickle.dump((self.language, self.userId, self.generator,
                         self.selector, self.ranker), pf,
                        pickle.HIGHEST_PROTOCOL)

    def load(self, userId=None):
        if not userId:
            userId = self.userId
        with open(LEXICAL_MODEL_PATH_TEMPLATE.format(userId), 'rb') as pf:
            unpickled = pickle.load(pf)
            logger.debug(unpickled)
            (self.language, self.userId, self.generator,
             self.selector, self.ranker) = unpickled
        return self

    def fresh_train(self, resources=None):
        if not resources:
            try:
                resources = RESOURCES[self.language]
            except KeyError:
                logger.error("Couldn't find resources for language "
                             "ID {}".format(self.language))
        # General purpose
        w2vpm = resources['embeddings']
        # Generator
        # gg = LexensteinGenerator(w2vpm)
        # gg = SynonymDBGenerator(w2vpm, resources['synonyms'])
        gg = LexensteinGenerator(w2vpm)

        # Selector
        fe = FeatureEstimator()
        fe.resources[w2vpm[0]] = gg.model
        fe.addCollocationalFeature(resources['lm'], 2, 2, 'Complexity')
        fe.addWordVectorSimilarityFeature(w2vpm[0], 'Simplicity')
        br = BoundaryRanker(fe)
        bs = BoundarySelector(br)
        bs.trainSelectorWithCrossValidation(resources['ubr'], 1, 5, 0.25,
                                            k='all')
        # Ranker
        fe = FeatureEstimator()
        fe.addLengthFeature('Complexity')
        fe.addCollocationalFeature(resources['lm'], 2, 2, 'Simplicity')
        orr = OnlineRegressionRanker(fe, None, training_dataset=resources[
                                         'ranking_training_dataset'])
        # Return LexicalSimplifier object
        self.generator = gg
        self.selector = bs
        self.ranker = orr
        return self

    def check_featurizer_set(self):
        return True


class DummyLexicalClassifier(Classifier):
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

    def predict_text(self, txt, ranker=None):
        """
        :param txt:
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
