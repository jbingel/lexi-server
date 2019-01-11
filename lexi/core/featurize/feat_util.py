from collections import defaultdict

import networkx as nx
import numpy as np
import spacy
from networkx.algorithms.traversal.depth_first_search import dfs_edges

from lexi.core.featurize.util import resources

COMMA = ","
VERB = "V"
nlp = spacy.load('en')


class EtymWN:

    def __init__(self):
        self.etymwn = None

    def retrieve_etymology(self, word, lang="eng"):
        if not self.etymwn:
            self.etymwn = resources.init_etymwn()
        try:
            word_etym = [edge[1] for edge in dfs_edges(self.etymwn, lang + ':' + word)]
        except KeyError:
            # print("Warning! Could not retrieve etymology for word '%s' in
            # language '%s'" %(word, lang))
            word_etym = []
        return word_etym


class LM:

    def __init__(self):
        self.lm_reg = None

    def prob(self, item, level="words", corpus="wp", order=1):
        if not self.lm_reg:
            self.lm_reg = resources.init_lms()
        if level == "chars":
            item = " ".join([c for c in item])
        p = 0
        try:
            lm = self.lm_reg[level][corpus]
            scorefunc = {1: lm.score_ug, 2: lm.score_bg, 3: lm.score_tg}
            try:
                p = scorefunc[order](item)
            except KeyError:
                print("Warning! No entry for item '%s' in language model for "
                      "level '%s' and corpus '%s'" %
                      (item, level, corpus))
        except KeyError:
            print("Error! Could not find language model for level '%s' and "
                  "corpus '%s'" % (level, corpus))
        return max(p, 0.00001)

    # bigram markov chain
    def seq_prob(self, target, conditional_seq):
        cond_prob = self.prob(conditional_seq[0])
        for i in range(len(conditional_seq) - 1):
            cond_prob *= self.prob(" ".join(conditional_seq[i:i + 2]), order=2)
        return cond_prob * self.prob(target)


def dep_head_of(sent, n):
    for u, v in sent.edges():
        if v == n:
            return u
    return None


def dep_pathtoroot(sent, child):
    return nx.predecessor(sent, child)


def count_vowels(word):
    vowels = "aeiouAEIOU"
    return sum(word.count(v) for v in vowels)


def commas_before_after(sent, idx):
    before = sum(1 for w in sent["lemma"][:idx] if w == COMMA)
    after = sum(1 for w in sent["lemma"][idx+1:] if w == COMMA)
    return before, after


def verbs_before_after(sent, idx):
    before = sum(1 for w in sent["pos"][:idx] if w.startswith(VERB))
    after = sum(1 for w in sent["pos"][idx+1:] if w.startswith(VERB))
    return before, after


def lbl2index(labels):
    label_set = set(labels)
    n = len(label_set)
    label_dict = {lbl: idx for (lbl, idx) in zip(label_set, range(n))}

    ys = []
    for l in labels:
        y = label_dict[l]
        ys.append(y)
    return np.array(ys, dtype='int')


def onehot_labels(labels):
    label_set = set(labels)
    n = len(label_set)
    label_dict = {lbl: idx for (lbl, idx) in zip(label_set, range(n))}

    ys = []
    for l in labels:
        y = np.zeros(n)
        y[label_dict[l]] = 1
        ys.append(y)
    return np.array(ys)


def has_ancestor_in_lang(lang, word_etym):
    for ancestor in word_etym:
        if ancestor.split(':')[0] == lang:
            return True
    return False


def read_sentences_plain(raw_data):
    doc = nlp(raw_data)
    words_seen = 0
    for s in doc.sents:
        sent = defaultdict(list)
        for i, w in enumerate(s):
            sent["idx"].append(i+1)
            sent["form"].append(w.text)
            sent["lemma"].append(w.lemma_)
            sent["pos"].append(w.pos_)
            ne = w.ent_type_ if w.ent_type_ else "O"
            sent["ne"].append(ne)
            # target = w.head.i - words_seen if w.dep_.lower() != "root" else -1
            target = w.head.i - words_seen
            sent["head"].append(target+1)
            sent["deprel"].append(w.dep_)
            sent["label"].append("?")
        words_seen += len(s)
        yield sent


def read_sentences(data):
    sent = defaultdict(list)
    # 0    In    in    IN    O    4    case    -
    for line in data:
        line = line.strip()
        if not line:
            yield(sent)
            sent = defaultdict(list)
        elif line.startswith("#"):
            pass
        else:
            fields = line.split("\t")
            if len(fields) == 8:
                idx, form, lemma, pos, ne, head, deprel, label = fields
            elif len(fields) == 7:
                idx, form, lemma, pos, ne, head, deprel = fields
                label = "?"
            else:
                raise ValueError("Line is malformed, splitting at tab does not "
                                 "yield 7 or 8 fields: " + line)
            sent["idx"].append(int(idx))
            sent["form"].append(form)
            sent["lemma"].append(lemma)
            sent["pos"].append(pos)
            sent["ne"].append(ne)
            sent["head"].append(head)
            sent["deprel"].append(deprel)
            sent["label"].append(label)

    if sent["idx"]:
        yield(sent)
