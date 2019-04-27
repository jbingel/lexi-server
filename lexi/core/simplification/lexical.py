import logging
import pickle
import os
import jsonpickle
import torch

from lexi.config import LEXICAL_MODEL_PATH_TEMPLATE, RANKER_MODEL_PATH_TEMPLATE
from lexi.core.simplification import SimplificationPipeline
from lexi.core.simplification.util import make_synonyms_dict, \
    parse_embeddings
from lexi.core.featurize.featurizers import LexicalFeaturizer
from lexi.core.util import util
from abc import ABCMeta, abstractmethod
import keras
from keras.layers import Input, Dense
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler

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
                    ranking = ranker.rank(candidates, sent, wb, we)
                elif self.ranker:
                    ranking = self.ranker.rank(candidates, sent, wb, we)
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


# class LexiFeaturizer(DictVectorizer):
#
#     def __init__(self):
#         super().__init__()
#
#     def dimensions(self):
#         return len(self.get_feature_names())
#         # return 3
#
#     def featurize(self, sentence, startOffset, endOffset):
#         featuredict = dict()
#         featuredict["word_length"] = endOffset - startOffset
#         featuredict["sentence_length"] = len(sentence)
#         self.transform(featuredict)
#
#     def save(self, path):
#         json = jsonpickle.encode(self)
#         with open(path, "w") as jsonfile:
#             jsonfile.write(json)
#
#     @staticmethod
#     def staticload(path):
#         with open(path) as jsonfile:
#             json = jsonfile.read()
#         return jsonpickle.decode(json)


class LexiCWI(LexiPersonalizedPipelineStep):

    def __init__(self, userId, featurizer=None):
        # self.model = self.build_model()
        super().__init__(userId)
        self.featurizer = featurizer if featurizer is not None else \
            LexiFeaturizer()
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def build_model(self):
        return LexiScorerNet(self.featurizer.dimensions(), [10, 10])

    def fresh_train(self, cwi_data):
        x, y = cwi_data
        self.model.fit(x, y, self.optimizer)

    def update(self, cwi_data):
        x, y = cwi_data
        self.model.fit(x, y, self.optimizer)  # TODO updating like this is problematic if we
        # want learning rate decay or other things that rely on previous
        # iterations, those are not saved in the model or optimizer...

    def identify_targets(self, sent, token_offsets):
        return [(wb, we) for wb, we in token_offsets if
                self.is_complex(sent, wb, we)]

    def is_complex(self, sent, startOffset, endOffset):
        x = self.featurizer.featurize(sent, startOffset, endOffset)
        logger.debug(x)
        cwi_score = self.model(x)
        return cwi_score > 0


class LexiFeaturizer(DictVectorizer):

    def __init__(self):
        super().__init__(sparse=False)
        self.scaler = MinMaxScaler()

    def dimensions(self):
        if hasattr(self, "feature_names_"):
            return len(self.get_feature_names())
        else:
            logger.warning("Asking for vectorizer dimensionality, "
                           "but vectorizer has not been fit yet. Returning 0.")
            return 0

    def to_dict(self, sentence, startOffset, endOffset):
        featuredict = dict()
        featuredict["word_length"] = endOffset - startOffset
        featuredict["sentence_length"] = len(sentence)
        return featuredict

    def fit(self, words_in_context):
        wic_dicts = [self.to_dict(*wic) for wic in words_in_context]
        vecs = super().fit_transform(wic_dicts)
        self.scaler.fit(vecs)

    def featurize(self, sentence, startOffset, endOffset, scale=True):
        vecs = self.transform(self.to_dict(sentence, startOffset, endOffset))
        if scale:
            vecs = self.scaler.transform(vecs)
        return vecs

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

    def __init__(self, userId, featurizer=None):
        super().__init__(userId)
        self.featurizer = featurizer or LexiFeaturizer()
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def build_model(self):
        return LexiScorerNet(self.featurizer.dimensions(), [10, 10])

    def fresh_train(self, data):
        x, y = data
        self.model.fit(x, y, self.optimizer)

    def update(self, cwi_data):
        x, y = cwi_data
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        self.model.fit(x, y, self.optimizer)  # TODO updating like this is
        # problematic if we want learning rate decay or other things that rely
        # on previous iterations, those are not saved in the model or optimizer...

    def set_featurizer(self, featurizer):
        self.featurizer = featurizer

    def rank(self, candidates, sentence=None, wb=0, we=0):
        scored_candidates = []
        for candidate in candidates:
            modified_sentence = sentence[:wb] + candidate + sentence[we:]
            x = self.featurizer.featurize(modified_sentence, wb,
                                          wb + len(candidate))
            score = self.model.forward(x)
            scored_candidates.append((candidate, score))
            logger.debug("Sorted candidates: {}".format(scored_candidates))
        return [candidate for candidate, score in sorted(scored_candidates,
                                                         key=lambda x: x[1])]

    def save(self, userId):
        json = jsonpickle.encode(self)
        with open(RANKER_MODEL_PATH_TEMPLATE.format(userId), 'w') as jsonfile:
            jsonfile.write(json)

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        return jsonpickle.decode(json)

    def train(self, data, batch_size=64, lr=1e-3,
                    epochs=30, dev=None, clip=None, early_stopping=None,
                    l2=1e-5, lr_schedule=None):

        loss = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                     weight_decay=l2)
        for input1, input2 in data:
            pass  # TODO


class LexiScorerNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(LexiScorerNet, self).__init__()
        self.input = torch.nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = [torch.nn.Linear(hidden_sizes[i],
                                              hidden_sizes[i+1])
                              for i in range(len(hidden_sizes)-1)]
        self.out = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        x = torch.Tensor(x)
        h = torch.relu(self.input(x))
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        return self.out(h)

    def fit(self, x, y, optimizer, epochs=1):
        for _ in range(epochs):
            self.train()
            # optimizer.zero_grad()
            pred = self.forward(x)
            # loss = torch.sqrt(torch.mean((y - pred) ** 2))
            loss = torch.mean((y - pred))
            loss.backward()
            optimizer.step()


class RankerNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(RankerNet, self).__init__()
        self.input = torch.nn.Linear(input_size, hidden_sizes[0])
        self.out = torch.nn.Linear(hidden_sizes[0] * 2, 1)

    def forward(self, input1, input2):
        l = self.input(torch.Tensor(input1))
        r = self.input(torch.Tensor(input2))
        combined = torch.cat((l.view(-1), r.view(-1)))
        return self.out(combined)



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
