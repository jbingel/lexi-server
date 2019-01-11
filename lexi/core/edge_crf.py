import argparse
import os
import pickle

import networkx as nx
import numpy as np
from networkx.algorithms.traversal import dfs_tree
from nltk.tokenize.moses import MosesDetokenizer
from pystruct.learners import FrankWolfeSSVM, SubgradientSSVM
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.utils import SaveLogger
from sklearn.metrics import f1_score, recall_score, precision_score, \
    accuracy_score

from lexi.core.featurize.feat_util import edge_featurize

scriptdir = os.path.dirname(os.path.realpath(__file__))
model_file = scriptdir + "/../models/{}.pickle"
detokenizer = MosesDetokenizer()


class EdgeCRFClassifier:
    def __init__(self, userId="anonymous"):
        self.model = None
        self.learner = None
        self.featurizer = None
        self.userId = userId

    def fresh_train(self, x, y, iterations=10):
        self.model = EdgeFeatureGraphCRF(inference_method="max-product")
        self.learner = SubgradientSSVM(model=self.model, max_iter=iterations,
                                       logger=SaveLogger(model_file.format(
                                           self.userId + "-learner")))
        self.learner.fit(x, y, warm_start=False)
        self.save()

    def fresh_train_default(self, iterations=10):
        default_train = scriptdir + '/../../../data/compression/' \
                                    'googlecomp100.train.lbl'
        featurizer = edge_featurize.Featurizer()
        x, y = featurizer.fit_transform(default_train)
        self.fresh_train(x, y, iterations=iterations)

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
        with open(model_file.format(userId), 'wb') as pf:
            pickle.dump((self.learner, self.model, self.featurizer), pf,
                        pickle.HIGHEST_PROTOCOL)

    def load(self, userId=None):
        if not userId:
            userId = self.userId
        with open(model_file.format(userId), 'rb') as pf:
            self.learner, self.model, self.featurizer = pickle.load(pf)
        return self

    def load_default_init(self):
        with open(model_file.format("default"), 'rb') as pf:
            self.learner, self.model, self.featurizer = pickle.load(pf)

    def check_featurizer_set(self):
        if not self.featurizer:
            raise RuntimeError("Featurizer not set. Use set_featurizer().")

    def text_predict(self, input_txt):
        original = []
        simplified = []
        X, parses = self.featurizer.transform_plain(input_txt)

        for x, parse in zip(X, parses):
            labels = self.predict([x])[0]
            # tokens = parses[0]['form']
            tokens = parse['form']
            original.append(detokenizer.detokenize([t for t in tokens], True))
            # original.append(" ".join([t for t in tokens]))
            # print('#\n#\n#')
            # print(" ".join(tokens) + "\t===>\t", end='')
            graph = nx.DiGraph()
            for s, t in x[1]:
                # graph.add_edge(tokens[s], tokens[t])
                graph.add_edge(s, t)

            # print(graph.nodes())
            for i, l in enumerate(labels[0]):
                if l == 'DEL':
                    for s, t in graph.edges():
                        # print(t, s)
                        if t == i:
                            # print("DEL", t)
                            for n in dfs_tree(graph, t).nodes():
                                # print(n)
                                graph.remove_node(n)

            # print(graph.nodes())
            simplified.append(detokenizer.detokenize(
                [tokens[n] for n in sorted(graph.nodes())], True))
            # simplified.append(" ".join(
            # [tokens[n] for n in sorted(graph.nodes())]))
        return original, simplified


# # # BELOW IS FOR DEV ONLY # # #


# def optimize_threshold(preds, gold):
#     best_t = 0
#     best_f1 = -1
#     t_results = {}
#     for thelp in range(1000):
#         t = thelp/1000.0
#         t_results[t] = eval_for_threshold(preds, gold, t)
#         f1 = t_results[t][0]
#         if f1 > best_f1:
#             best_t = t
#             best_f1 = f1
#     return best_t, t_results[best_t]


def evaluate(pred, gold):
    r = recall_score(gold, pred, average="micro", labels=[1, 2])
    p = precision_score(gold, pred, average="micro", labels=[1, 2])
    f1 = f1_score(gold, pred, average="micro", labels=[1, 2])
    a = accuracy_score(gold, pred)
    return f1, r, p, a


# def eval_for_threshold(scores, gold, t):
#     pred = [0 if (score < t) else 1 for score in scores]
#     return evaluate(pred, gold)
#
#
# def test(model, X_test, y_test, t=None):
#     if t:
#         preds = model.predict(X_test)
#         results = eval_for_threshold(preds, y_test, t)
#     else:
#         scores = model.predict(X_test)
#         t, results = optimize_threshold(scores, y_test)
#     return t, results


def load_pickled(data):
    with open(data, 'rb') as pickled_data:
        X, y = pickle.load(pickled_data)
    print("Loaded data with %d instances" % (len(X)))
    return X, y


# def crossval(X,y,splits, layers=[150,50], iterations=10, t=None):
#     results = []
#     ts = []
#     m = len(X)
#     cs = [(i*m/splits, (i+1)*len(X)/splits) for i in range(splits)]
#     for s,e in cs:
#         X_tr = np.array([X[i] for i in range(m) if i < s or i >= e])
#         X_te = np.array([X[i] for i in range(m) if i >= s and i < e])
#         y_tr = np.array([y[i] for i in range(m) if i < s or i >= e])
#         y_te = np.array([y[i] for i in range(m) if i >= s and i < e])
#
#         # nn = NN(conf)
#         # nn.train(X_tr, y_tr, conf.iterations)
#         nn = trainModel(X_tr, y_tr, layers, iterations)
#         best_t, res = test(nn, X_te, y_te, t)
#         ts.append(best_t)
#         results.append(res)
#
#     f1s = [res[0] for res in results]
#     rec = [res[1] for res in results]
#     acc = [res[2] for res in results]
#     pre = [res[3] for res in results]
#
#     print('\nF1  | {:.3f}   (std {:.3f})'.format(np.average(f1s), np.std(f1s)))
#     print('Rec | {:.3f}   (std {:.3f})'.format(np.average(rec), np.std(rec)))
#     print('Acc | {:.3f}   (std {:.3f})'.format(np.average(acc), np.std(acc)))
#     print('Pre | {:.3f}   (std {:.3f})'.format(np.average(pre), np.std(pre)))
#
#     return ts, f1s

#
# def combine_data(train_path, test_path, out_path):
#     cutoff = 0
#     with open(out_path, 'w') as outFile:
#         with open(train_path, 'r') as tr:
#             lines = tr.readlines()
#             for l in lines:
#                 if len(l.strip()) <= 1:
#                     cutoff += 1  # counts no of sentences (=empty lines) in train
#                 outFile.write(l)
#         with open(test_path, 'r') as te:
#             outFile.write(te.read())
#     return cutoff


def main():
    default_train = \
        scriptdir+'/../../../data/compression/googlecomp100.train.lbl'
    default_test = \
        scriptdir+'/../../../data/compression/googlecomp.dev.lbl'
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', '-t', type=float,
                        help='Threshold for predicting 0/1. ')
    parser.add_argument('--iterations', '-i', type=int, default=50,
                        help='Training iterations.')
    parser.add_argument('--data', '-d', default=default_train,
                        help='Features and labels')
    parser.add_argument('--testdata', default=default_test,
                        help='Test data (not needed for crossval).')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true',
                        help='Print avg. loss at every iter.')
    parser.add_argument('--output', '-o', help="Output file")
    parser.add_argument('--features', '-f', dest='features', default=[],
                        type=str, nargs='+', help='Used feature types')
    parser.add_argument('--train', action='store_true',
                        help='If set, will train the model')

    args = parser.parse_args()

    featurizer = edge_featurize.Featurizer(args.features)
    X, y = featurizer.fit_transform(default_train)

    crf = EdgeFeatureGraphCRF(inference_method="max-product")
    model = FrankWolfeSSVM(model=crf, C=.1, max_iter=args.iterations)
    model.fit(X, y)
    if args.testdata:
        X_te, y_te = featurizer.transform(args.testdata)
        pred = model.predict(X_te)
        pred_flat = [item for sublist in pred for item in sublist]
        y_te_flat = [item for sublist in y_te for item in sublist]
        if args.output:
            with open(args.output, 'w') as of:
                for sent_pred in pred:
                    for lid in sent_pred:
                        # print(lid)
                        of.write('%s\n' % featurizer.mapper.id2label[lid])
                    of.write('\n')
        res = evaluate(pred_flat, y_te_flat)
        resout = "F1: %f, R: %f, A: %f, P: %f\n" % res
        print(resout)

if __name__ == '__main__':
    main()
