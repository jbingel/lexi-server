import numpy as np
from sklearn.feature_extraction import DictVectorizer

from lexi.core.featurize import extract_lexical_feats, feat_util
from lexi.core.featurize.extract_sentence_feats import TreeNode
from abc import ABCMeta, abstractmethod


class LabelMapper:
    def __init__(self):
        self.label2id = dict()
        self.id2label = {}

    def map_batch(self, labels):
        out = []
        for label in labels:
            out.append(self.map(label))
        return out

    def map(self, label):
        if label not in self.label2id:
            newId = len(self.label2id)
            self.label2id[label] = newId
            self.id2label[newId] = label
        return self.label2id[label]

    def map_inv(self, ids):
        out = []
        for _id in ids:
            out.append(self.id2label.get(_id, "?"))
        return out


class LexiFeaturizer(metaclass=ABCMeta):

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError


class Featurizer:

    def __init__(self, features=None):
        self.mapper = LabelMapper()
        self.features = features

    def fit_transform(self, data):
        return self.transform(data, fit=True)

    def transform(self, data, fit=False):
        raise NotImplementedError

    def transform_plain(self, data):
        raise NotImplementedError

    def map_labels(self, data):
        return self.mapper.map_batch(data)

    def map_inv(self, ids):
        return self.mapper.map_inv(ids)


class LexicalFeaturizer(Featurizer):

    def __init__(self, features=None):
        super().__init__(features)
        self.vectorizer = DictVectorizer()

    def featurize_word(self, w):
        word = extract_lexical_feats.Word(w)
        return word.featurize_by_type(self.features)

    def transform(self, data, fit=False):
        feats = []
        labels = []
        for word in data:
            feats.append(self.featurize_word(word))
            # labels.append(label)
        if fit:
            feats = self.vectorizer.fit_transform(feats)
        else:
            feats = self.vectorizer.transform(feats)

        # labels = np.array(labels)
        return feats, labels

    def transform_plain(self, data):
        return self.transform(data, fit=False)


class PystructEdgeFeaturizer(Featurizer):

    def __init__(self, features=None):
        super().__init__(features)
        self.node_vectorizer = DictVectorizer()
        self.edge_vectorizer = DictVectorizer()
        # self.nlp = spacy.load('en')
        print("Loaded natural language processor.")

    def prettyprintweights(self, linearmodel):
        for name, value in zip(self.node_vectorizer.feature_names_, linearmodel.coef_[0]):
            print("\t".join([name, str(value)]))

    def featurize_sentence(self, s):
        nodefeats = []
        edges = []
        edgefeats = []
        labels = []
        for l, i in zip(s["label"], s["idx"]):
            i -= 1
            w = TreeNode(s, i, s["form"][i], s["lemma"][i], s["pos"][i],
                         s["ne"][i], s["head"], s["deprel"], l)
            nodefeats.append(w.featurize_by_type(self.features))
            head = int(s["head"][i])
            tgt = head if head > 0 else i+1
            edges.append((tgt-1, i))
            edgefeats.append(w.featurize_by_type(["dependency"]))
            labels.append(l)
        return nodefeats, edges, edgefeats, labels

    def fit_transform(self, data):
        return self.transform(data, fit=True)

    def transform(self, data, fit=False):
        labels = []
        X = []
        y = []
        sentence_lengths = []
        Xnodefeats = []
        Xedges = []
        Xedgefeats = []
        print("Collecting features...")
        # for s in feat_util.read_sentences_plain(data):
        for s in feat_util.read_sentences(data):
            nodefeats, edges, edgefeats, nodelabels = self.featurize_sentence(s)
            sentence_lengths.append(len(nodefeats))
            Xnodefeats.extend(nodefeats)
            Xedges.extend(edges)
            Xedgefeats.extend(edgefeats)
            labels.extend(nodelabels)

        if fit:
            Xnodefeats = self.node_vectorizer.fit_transform(Xnodefeats).toarray()
            Xedgefeats = self.edge_vectorizer.fit_transform(Xedgefeats).toarray()
        else:
            Xnodefeats = self.node_vectorizer.transform(Xnodefeats).toarray()
            Xedgefeats = self.edge_vectorizer.transform(Xedgefeats).toarray()
        i = 0
        for sl in sentence_lengths:
            X.append((Xnodefeats[i:i+sl], np.array(Xedges[i:i+sl]), Xedgefeats[i:i+sl]))
            y.append(np.array(self.mapper.map_batch(labels[i:i + sl])))
            i = i+sl

        for i in range(len(X)):
            if not len(X[i][0]) == len(y[i]):
                print("unequal {}: {} vs {}".format(i, len(X[i][0]), len(y[i])))
        return X, y

    def transform_plain(self, data):
        X = []
        parses = []
        sentence_lengths = []
        Xnodefeats = []
        Xedges = []
        Xedgefeats = []
        print("Collecting features...")
        for s in feat_util.read_sentences_plain(data):
            nodefeats, edges, edgefeats, _ = self.featurize_sentence(s)
            sentence_lengths.append(len(nodefeats))
            Xnodefeats.extend(nodefeats)
            Xedges.extend(edges)
            Xedgefeats.extend(edgefeats)
            parses.append(s)

        Xnodefeats = self.node_vectorizer.transform(Xnodefeats).toarray()
        Xedgefeats = self.edge_vectorizer.transform(Xedgefeats).toarray()
        i = 0
        for sl in sentence_lengths:
            X.append((Xnodefeats[i:i+sl], np.array(Xedges[i:i+sl]), Xedgefeats[i:i+sl]))
            i = i+sl
        return X, parses


class PystructChainFeaturizer(Featurizer):

    def __init__(self, features=None):
        super().__init__(features)
        self.node_vectorizer = DictVectorizer()
        self.edge_vectorizer = DictVectorizer()
        # self.nlp = spacy.load('en')
        print("Loaded natural language processor.")

    def prettyprintweights(self, linearmodel):
        for name, value in zip(self.node_vectorizer.feature_names_, linearmodel.coef_[0]):
            print("\t".join([name, str(value)]))

    def featurize_sentence(self, s):
        nodefeats = []
        labels = []
        for l, i in zip(s["label"], s["idx"]):
            i -= 1
            w = TreeNode(s, i, s["form"][i], s["lemma"][i], s["pos"][i],
                         s["ne"][i], s["head"], s["deprel"], l)
            nodefeats.append(w.featurize_by_type(self.features))
            labels.append(l)
        return nodefeats, labels

    def fit_transform(self, data):
        return self.transform(data, fit=True)

    def transform(self, data, fit=False):
        labels = []
        X = []
        y = []
        sentence_lengths = []
        Xnodefeats = []
        Xedges = []
        Xedgefeats = []
        print("Collecting features...")
        # for s in feat_util.read_sentences_plain(data):
        for s in feat_util.read_sentences(data):
            nodefeats, nodelabels = self.featurize_sentence(s)
            sentence_lengths.append(len(nodefeats))
            Xnodefeats.extend(nodefeats)
            labels.extend(nodelabels)

        if fit:
            Xnodefeats = self.node_vectorizer.fit_transform(Xnodefeats).toarray()
            Xedgefeats = self.edge_vectorizer.fit_transform(Xedgefeats).toarray()
        else:
            Xnodefeats = self.node_vectorizer.transform(Xnodefeats).toarray()
            Xedgefeats = self.edge_vectorizer.transform(Xedgefeats).toarray()
        i = 0
        for sl in sentence_lengths:
            X.append((Xnodefeats[i:i+sl], np.array(Xedges[i:i+sl]), Xedgefeats[i:i+sl]))
            y.append(np.array(self.mapper.map_batch(labels[i:i + sl])))
            i = i+sl

        for i in range(len(X)):
            if not len(X[i][0]) == len(y[i]):
                print("unequal {}: {} vs {}".format(i, len(X[i][0]), len(y[i])))
        return X, y

    def transform_plain(self, data):
        X = []
        parses = []
        sentence_lengths = []
        Xnodefeats = []
        print("Collecting features...")
        for s in feat_util.read_sentences_plain(data):
            nodefeats, _ = self.featurize_sentence(s)
            sentence_lengths.append(len(nodefeats))
            Xnodefeats.extend(nodefeats)
            parses.append(s)

        Xnodefeats = self.node_vectorizer.transform(Xnodefeats).toarray()
        i = 0
        for sl in sentence_lengths:
            X.append(Xnodefeats[i:i+sl])
            i = i+sl
        return X, parses
