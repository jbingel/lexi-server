import logging
import pickle
import jsonpickle
import torch

from lexi.config import LEXICAL_MODEL_PATH_TEMPLATE, RANKER_PATH_TEMPLATE, \
    SCORER_PATH_TEMPLATE, SCORER_MODEL_PATH_TEMPLATE, CWI_PATH_TEMPLATE
from lexi.core.simplification import SimplificationPipeline
from lexi.core.simplification.util import make_synonyms_dict, \
    parse_embeddings
from lexi.core.featurize.featurizers import LexicalFeaturizer
from lexi.core.util import util
from abc import ABCMeta, abstractmethod

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

    def __init__(self, userId=None, scorer=None):
        self.userId = userId
        self.scorer = scorer
        self.scorer_path = None

    def set_scorer(self, scorer):
        self.scorer = scorer
        self.scorer_path = scorer.get_path()

    def set_userId(self, userId):
        self.userId = userId

    @abstractmethod
    def update(self, data):
        raise NotImplementedError

    def __getstate__(self):
        """
        Needed to save pipeline steps using jsonpickle, since this module cannot
        handle torch models -- we use torch's model saving functionality
        instead. This is the method used by jsonpickle to get the state of the
        object when serializing.
        :return:
        """
        state = self.__dict__.copy()
        del state['scorer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class LexiCWI(LexiPersonalizedPipelineStep):

    def __init__(self, userId, scorer=None):
        super().__init__(userId, scorer)
        self.cwi_threshold = 0.67

    def identify_targets(self, sent, token_offsets):
        return [(wb, we) for wb, we in token_offsets if
                self.is_complex(sent, wb, we)]

    def is_complex(self, sent, startOffset, endOffset):
        cwi_score = self.scorer.score(sent, startOffset, endOffset)
        return cwi_score > self.cwi_threshold

    def set_cwi_threshold(self, threshold):
        self.cwi_threshold = threshold

    def update(self, data):
        if self.scorer:
            self.scorer.update(data)

    def save(self, userId):
        json = jsonpickle.encode(self)
        with open(CWI_PATH_TEMPLATE.format(userId), 'w') as jsonfile:
            jsonfile.write(json)

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        cwi = jsonpickle.decode(json)
        if hasattr(cwi, "scorer_path") and cwi.scorer_path is not None:
            cwi.set_scorer(LexiScorer.staticload(cwi.scorer_path))
        else:
            logger.warn("Ranker file does not provide link to a scorer. Set "
                        "manually with ranker.set_scorer()!")
        return cwi


class LexiRanker(LexiPersonalizedPipelineStep):

    def __init__(self, userId, scorer=None):
        super().__init__(userId, scorer)

    def update(self, data):
        if self.scorer is not None:
            self.scorer.update(data)

    def rank(self, candidates, sentence=None, wb=0, we=0):
        scored_candidates = []
        for candidate in candidates:
            modified_sentence = sentence[:wb] + candidate + sentence[we:]
            score = self.scorer.score(modified_sentence, wb,
                                      wb + len(candidate))
            scored_candidates.append((candidate, score))
            logger.debug("Sorted candidates: {}".format(scored_candidates))
        return [candidate for candidate, score in sorted(scored_candidates,
                                                         key=lambda x: x[1])]

    def save(self, userId):
        json = jsonpickle.encode(self)
        with open(RANKER_PATH_TEMPLATE.format(userId), 'w') as jsonfile:
            jsonfile.write(json)
        self.scorer.save()

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        ranker = jsonpickle.decode(json)
        if hasattr(ranker, "scorer_path") and ranker.scorer_path is not None:
            ranker.set_scorer(LexiScorer.staticload(ranker.scorer_path))
        else:
            logger.warn("Ranker file does not provide link to a scorer. Set "
                        "manually with ranker.set_scorer()!")
        return ranker


class LexiScorer:
    def __init__(self, userId, featurizer, hidden_dims):
        self.userId = userId
        self.path = SCORER_PATH_TEMPLATE.format(userId)
        self.featurizer = featurizer
        self.hidden_dims = hidden_dims
        self.model = self.build_model()
        self.model_path = SCORER_MODEL_PATH_TEMPLATE.format(self.userId)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.update_steps = 0
        self.cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model'], state['cache']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache = {}
        self.model = self.build_model()

    def get_path(self):
        return SCORER_PATH_TEMPLATE.format(self.userId)

    def set_userId(self, userId):
        self.userId = userId
        self.path = SCORER_PATH_TEMPLATE.format(userId)

    def build_model(self):
        return LexiScorerNet(self.featurizer.dimensions(), self.hidden_dims)

    def train_model(self, x, y):
        self.model.fit(torch.Tensor(x), torch.Tensor(y), self.optimizer)

    def update(self, data):
        # TODO do this in one batch (or several batches of more than 1 item...)
        for (sentence, start_offset, end_offset), label in data:
            x = self.featurizer.featurize(sentence, start_offset, end_offset)
            self.model.fit(x, label, self.optimizer)
            self.update_steps += 1

    def score(self, sent, start_offset, end_offset):
        cached = self.cache.get((sent, start_offset, end_offset))
        if cached is not None:
            return cached
        self.model.eval()
        x = self.featurizer.featurize(sent, start_offset, end_offset)
        score = float(self.model.forward(x))
        self.cache[(sent, start_offset, end_offset)] = score
        return score

    def save(self):
        # save state of this object, except model (excluded in __getstate__())
        with open(self.get_path(), 'w') as f:
            json = jsonpickle.encode(self)
            f.write(json)
        # save model
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, self.model_path)

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        scorer = jsonpickle.decode(json)
        scorer.cache = {}
        scorer.model = scorer.build_model()
        checkpoint = torch.load(scorer.model_path)
        scorer.model.load_state_dict(checkpoint['model_state_dict'])
        return scorer


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

    def fit(self, x, y, optimizer, epochs=100):
        for _ in range(epochs):
            self.train()
            # optimizer.zero_grad()
            pred = self.forward(x)
            # loss = torch.sqrt(torch.mean((y - pred) ** 2))
            loss = torch.mean((y - pred))
            print(loss)
            loss.backward()
            optimizer.step()


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
