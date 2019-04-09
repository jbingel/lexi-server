import logging
import pickle
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
from networkx import dfs_tree
from pystruct.learners import SubgradientSSVM
from pystruct.models import ChainCRF, EdgeFeatureGraphCRF
from pystruct.utils import SaveLogger

from lexi.core.util import util
from lexi.core.featurize.featurizers import PystructChainFeaturizer, \
    PystructEdgeFeaturizer
from lexi.config import MODEL_PATH_TEMPLATE
from lexi.core.simplification import SimplificationPipeline
from lexi.core.simplification import detokenizer

logger = logging.getLogger('lexi')


class PystructSimplificationPipeline(SimplificationPipeline, metaclass=ABCMeta):
    def __init__(self, userId="anonymous"):
        self.model = None
        self.learner = None
        self.featurizer = None
        self.userId = userId

    @abstractmethod
    def fresh_train(self, x, y):
        pass

    def update(self, x, y):
        """
        Performs an online update of the model
        :param x: Input data
        :param y: List of Numpy array of label IDs
        :return:
        """
        self.learner.fit(x, y, warm_start=False)

    def predict(self, x):
        self.check_featurizer_set()
        label_ids = self.learner.predict(x)
        labels = []
        for sent in label_ids:
            labels.append(np.array(self.featurizer.map_inv(sent)))
        return labels, label_ids

    def set_featurizer(self, featurizer):
        self.featurizer = featurizer

    def featurize_train(self, train_data, iterations=10):
        self.check_featurizer_set()
        x, y = self.featurizer.fit_transform(train_data)
        self.fresh_train(x, y, iterations)

    def featurize_update(self, src, y):
        self.check_featurizer_set()
        x, _ = self.featurizer.transform(src)
        self.update(x, y)

    def featurize_predict(self, data):
        self.check_featurizer_set()
        x, _ = self.featurizer.transform(data)
        return self.predict(x)

    def save(self, userId=None):
        if not userId:
            userId = self.userId
        with open(MODEL_PATH_TEMPLATE.format(userId), 'wb') as pf:
            pickle.dump((self.learner, self.model, self.featurizer), pf,
                        pickle.HIGHEST_PROTOCOL)

    def load(self, userId=None):
        if not userId:
            userId = self.userId
        with open(MODEL_PATH_TEMPLATE.format(userId), 'rb') as pf:
            self.learner, self.model, self.featurizer = pickle.load(pf)
        return self

    def load_default_init(self):
        with open(MODEL_PATH_TEMPLATE.format("default"), 'rb') as pf:
            self.learner, self.model, self.featurizer = pickle.load(pf)

    def simplify_text(self, input_txt):
        original = []
        simplified = []
        X, parses = self.featurizer.transform_plain(input_txt)
        for x, parse in zip(X, parses):
            if isinstance(self, EdgeCRFClassifier):
                preds = self.predict([x])[0]
            else:
                preds = self.predict([x[0]])[0]
            # tokens = parses[0]['form']
            tokens = parse['form']
            original.append(detokenizer.detokenize([t for t in tokens], True))
            # original.append(" ".join([t for t in tokens]))
            # logger.debug('#\n#\n#')
            # logger.debug(" ".join(tokens) + "\t===>\t", end='')
            graph = nx.DiGraph()
            for s, t in x[1]:
                # graph.add_edge(tokens[s], tokens[t])
                graph.add_edge(s, t)

            # logger.debug(graph.nodes())
            for i, l in enumerate(preds[0]):
                if l == 'DEL':
                    for s, t in graph.edges():
                        # logger.debug(t, s)
                        if t == i:
                            # logger.debug("DEL", t)
                            for n in dfs_tree(graph, t).nodes():
                                # logger.debug(n)
                                graph.remove_node(n)

            # logger.debug(graph.nodes())
            simplified.append(detokenizer.detokenize(
                [tokens[n] for n in sorted(graph.nodes())], True))
            # simplified.append(" ".join(
            # [tokens[n] for n in sorted(graph.nodes())]))
        return original, simplified


class ChainCRFClassifier(PystructSimplificationPipeline):

    def fresh_train(self, x, y, iterations=10):
        self.model = ChainCRF(inference_method="max-product")
        self.learner = SubgradientSSVM(
            model=self.model, max_iter=iterations,
            logger=SaveLogger(MODEL_PATH_TEMPLATE.format(self.userId + "-learner")),
            show_loss_every=50)
        self.learner.fit(x, y, warm_start=False)
        self.save()

    def check_featurizer_set(self):
        if not self.featurizer:
            featurizer = PystructChainFeaturizer()
            self.set_featurizer(featurizer)
            logger.debug("WARNING! Featurizer not set, setting new default "
                         "featurizer")

    # TODO provide default training data to first init model
    # def fresh_train_default(self, iterations=10):
    #     self.fresh_train(x, y, iterations=iterations)


class EdgeCRFClassifier(PystructSimplificationPipeline):

    def fresh_train(self, x, y, iterations=10, decay_rate=1):
        self.model = EdgeFeatureGraphCRF(inference_method="max-product")
        self.learner = SubgradientSSVM(
            model=self.model, max_iter=iterations,
            logger=SaveLogger(MODEL_PATH_TEMPLATE.format(self.userId + "-learner")),
            show_loss_every=50, decay_exponent=decay_rate)
        self.learner.fit(x, y, warm_start=False)
        self.save()

    def check_featurizer_set(self):
        if not self.featurizer:
            featurizer = PystructEdgeFeaturizer()
            self.set_featurizer(featurizer)
            logger.info("WARNING! Featurizer not set, setting new default "
                        "featurizer")

    # TODO provide default training data to first init model
    # def fresh_train_default(self, iterations=10):
    #     self.fresh_train(x, y, iterations=iterations)


class AveragedPerceptron(SimplificationPipeline):
    """
    An averaged perceptron, as implemented by Matthew Honnibal.

    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/
        a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    """

    def __init__(self, userId="anonymous"):
        self.userId = userId
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        self.classes = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
        # Do a secondary alphabetic sort, for stability
        return max(self.classes, key=lambda label: (scores[label], label))

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)
        return None

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights
        return None

    def save(self, userId=None):
        '''Save the pickled model weights.'''
        return pickle.dump(dict(self.weights), open(self.userId, 'w'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path))
        return None


class OnlineStructuredPerceptron(SimplificationPipeline):
    """Implements a first order CRF"""

    def __init__(self,
                 # observation_labels, state_labels,
                 learning_rate=1.0,
                 averaged=True):
        self.decoder = util.SequenceClassificationDecoder()
        self.observation_labels = None
        self.state_labels = None
        self.trained = False
        self.featurizer = None
        self.parameters = np.zeros(self.featurizer.get_num_features())
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []

    def set_featurizer(self, featurizer):
        self.featurizer = featurizer

    def fresh_train(self, x, y, iterations=10):
        self.parameters = np.zeros(self.featurizer.get_num_features())
        self.observation_labels = self.featurizer.observation_labels
        self.state_labels = self.featurizer.state_labels
        num_examples = len(x)
        for epoch in range(iterations):
            num_labels_total = 0
            num_mistakes_total = 0
            for i in range(num_examples):
                num_labels, num_mistakes = self.update(x[i], y[i])
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
            self.params_per_epoch.append(self.parameters.copy())
            acc = 1.0 - num_mistakes_total / num_labels_total
            logger.debug("Epoch: %i Accuracy: %f" % (epoch, acc))
        self.trained = True

        if self.averaged:
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w /= len(self.params_per_epoch)
            self.parameters = new_w

    def update(self, x, y):
        init_feats, trans_feats, final_feats, emit_feats = \
            self.featurizer.get_sequence_features(x)

        w = self.parameters

        pred = self.viterbi_decode(x)[0]
        num_mistakes = sum(y != pred.y)

        init_feats_p, trans_feats_p, final_feats_p, emit_feats_p = \
            self.featurizer.get_sequence_features(pred)

        if not y[0] == pred.y[0]:
            w[init_feats[0]] += 1
            w[init_feats_p[0]] -= 1

        for i in range(len(x) - 1):
            if not y[i] == pred.y[i]:
                w[trans_feats[i]] += 1
                w[trans_feats_p[i]] -= 1
                for f in emit_feats[i]:
                    w[f] += 1
                for f in emit_feats_p[i]:
                    w[f] -= 1

        if not y[-1] == pred.y[-1]:
            for f in emit_feats[-1]:
                w[f] += 1
            for f in emit_feats_p[-1]:
                w[f] -= 1
            w[final_feats[-1]] += 1
            w[final_feats_p[-1]] -= 1

        self.parameters = w
        return len(x), num_mistakes

    def viterbi_decode(self, sequence):
        """Compute the most likely sequence of states given the observations,
        by running the Viterbi algorithm."""

        # Compute scores given the observation sequence.
        initial_scores, transition_scores, final_scores, emission_scores = \
            self.compute_scores(sequence)

        # Run the forward algorithm.
        best_states, total_score = self.decoder.run_viterbi(
            initial_scores,
            transition_scores,
            final_scores,
            emission_scores)

        predicted_sequence = sequence.copy_sequence()
        predicted_sequence.y = best_states
        return predicted_sequence, total_score

    def compute_scores(self, sequence):
        num_states = self.get_num_states()
        length = len(sequence.x)
        emission_scores = np.zeros([length, num_states])
        initial_scores = np.zeros(num_states)
        transition_scores = np.zeros([length - 1, num_states, num_states])
        final_scores = np.zeros(num_states)

        # Initial position.
        for tag_id in range(num_states):
            initial_features = self.featurizer.get_initial_features(
                sequence, tag_id)
            score = 0.0
            for feat_id in initial_features:
                score += self.parameters[feat_id]
            initial_scores[tag_id] = score

        # Intermediate position.
        for pos in range(length):
            for tag_id in range(num_states):
                emission_features = self.featurizer.get_emission_features(
                    sequence, pos, tag_id)
                score = 0.0
                for feat_id in emission_features:
                    score += self.parameters[feat_id]
                emission_scores[pos, tag_id] = score
            if pos > 0:
                for tag_id in range(num_states):
                    for prev_tag_id in range(num_states):
                        transition_features = \
                            self.featurizer.get_transition_features(
                                sequence, pos, tag_id, prev_tag_id)
                        score = 0.0
                        for feat_id in transition_features:
                            score += self.parameters[feat_id]
                        transition_scores[
                            pos - 1, tag_id, prev_tag_id] = score

        # Final position.
        for prev_tag_id in range(num_states):
            final_features = self.featurizer.get_final_features(
                sequence, prev_tag_id)
            score = 0.0
            for feat_id in final_features:
                score += self.parameters[feat_id]
            final_scores[prev_tag_id] = score

        return initial_scores, transition_scores, final_scores, emission_scores

    def predict(self, x):
        return self.viterbi_decode(x)[0]

    def simplify_text(self, txt):
        # TODO
        pass

    def load_default_init(self):
        # TODO
        pass

    def save(self, userId=None):
        fn = open(userId, 'w')
        for p_id, p in enumerate(self.parameters):
            fn.write("%i\t%f\n" % (p_id, p))
        fn.close()

    def load(self, userId=None):
        fn = open(userId, 'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()