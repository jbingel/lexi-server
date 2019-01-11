import networkx as nx
import numpy as np
from nltk.corpus import wordnet as wn

from lexi.core.featurize import feat_util
from lexi.core.featurize.util import porter

lm = feat_util.LM()
etymwn = feat_util.EtymWN()


class Sentence:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def from_treenodes(nodes, tree):
        edges_from = []
        edges_to = []
        for e in tree.edges():
            src = e[0] if e[0] > 0 else e[1]
            edges_from.append(src-1)
            edges_to.append(e[1]-1)
        edges = np.asarray([np.array(edges_from), np.array(edges_to)])
        return np.array([nodes, edges])

    from_treenodes = staticmethod(from_treenodes)


class TreeNode:
    def __init__(self, sentence, index, word, lemma, pos, namedentity, heads, deprels, label):
        self.sentence = sentence  # sentence is a list of forms
        self.word = word
        self.index = int(index)
        self.label = label
        self.lemma = lemma
        self.pos = pos
        self.a_namedentity = namedentity
        self._heads = [int(h) for h in heads]  # "protected" property
        self._deprels = deprels                # "protected" property
        self.deptree = self._create_deptree()  # remember to use self.index+1 to access self.deptree

    def _create_deptree(self):
        deptree = nx.DiGraph()
        for idx_from_zero, head in enumerate(self._heads):
            # deptree.add_node(idx_from_zero+1) # the Id of each node is its Conll index (i.e. from one)
            # edge from head to dependent with one edge labeling called deprel
            deptree.add_edge(head, idx_from_zero+1, deprel=self._deprels[idx_from_zero])
        return deptree

    def a_simple_feats(self):
        d = dict()
        # d["a_form"] = self.word
        # d["a_lemma"] = self.lemma
        d["a_pos"] = self.pos
        d["a_namedentity"] = self.a_namedentity
        d["a_formlength"] = len(self.word)
        return d

    def a_simple_feats_lexicalized(self):
        d = dict()
        d["a_form"] = self.word
        d["a_lemma"] = self.lemma
        d["a_pos"] = self.pos
        d["a_namedentity"] = self.a_namedentity
        d["a_formlength"] = len(self.word)
        return d

    def b_wordnet_feats(self):
        d = dict()
        d["b_nsynsets"] = len(wn.synsets(self.word))
        return d

    def c_positional_feats(self):
        d = dict()
        d["c_relativeposition"] = int(self.index / len(self.sentence))
        before, after = feat_util.commas_before_after(self.sentence, self.index)
        d["c_preceding_commas"] = before
        d["c_following_commas"] = after
        before, after = feat_util.verbs_before_after(self.sentence, self.index)
        d["c_preceding_verbs"] = before
        d["c_following_verbs"] = after
        return d

    def d_frequency_feats(self):
        d = dict()
        wprob = max(lm.prob(self.word, corpus="wp"), 0.00001)
        wprobsimple = max(lm.prob(self.word, corpus="swp"), 0.00001)
        d["d_freq_in_swp"] = wprobsimple
        d["d_freq_in_wp"] = wprob
        d["d_freq_ratio_swp/wp"] = wprobsimple / wprob
        # TODO d["d_freqrank_distance_swp/wp"] = rank(self.word, corpus="swp") - rank(self.word, corpus="wp")
        # TODO d["d_distributional_distance_swp/wp"] = dist(dist_vector(self.word, "swp"),
        # dist_vector(self.word, "wp"))  # get distributional vector from background corpora, use some dist measure
        return d

    def e_morphological_feats(self):
        d = dict()
        etymology = etymwn.retrieve_etymology(self.lemma)
        d["e_latin_root"] = feat_util.has_ancestor_in_lang("lat", etymology)  # check wiktionary
        d["e_length_dist_lemma_form"] = len(self.word) - len(self.lemma)
        stem, steps = porter.stem(self.word)
        d["e_length_dist_stem_form"] = len(self.word) - len(stem)
        d["e_inflectional_morphemes_count"] = steps
        return d

    def f_prob_in_context_feats(self):
        d = dict()
        # d["f_P(w|w-1)"] = lm.seq_prob(self.word, [self.sentence[self.index-1]])
        # d["f_P(w|w-2w-1)"] = lm.seq_prob(self.word, [self.sentence[self.index-2:self.index]])
        # d["f_P(w|w+1)"] = lm.seq_prob(self.word, [self.sentence[self.index+1]])
        # d["f_P(w|w+1w+2)"] = lm.seq_prob(self.word, [self.sentence[self.index+1:self.index+3]])
        return d

    def g_char_complexity_feats(self):
        d = dict()
        prob_unigram = lm.prob(self.word, level="chars", order=1)
        prob_unigram_simple = lm.prob(self.word, level="chars", corpus="swp", order=1)
        prob_bigram = lm.prob(self.word, level="chars", order=2)
        prob_bigram_simple = lm.prob(self.word, level="chars", corpus="swp", order=2)
        d["g_char_unigram_prob"] = prob_unigram
        d["g_char_unigram_prob_ratio"] = prob_unigram_simple / prob_unigram
        d["g_char_bigram_prob"] = prob_bigram
        d["g_char_bigram_prob_ratio"] = prob_bigram_simple / prob_bigram
        d["g_vowels_ratio"] = float(feat_util.count_vowels(self.word)) / len(self.word)
        return d

    # def h_brownpath_feats(self):
    #     d = dict()
    #     #brown cluster path feature
    #     #global brownclusters
    #     if not feat_util.brownclusters:
    #         feat_util.init_brown()
    #     if self.word in feat_util.brownclusters:
    #         d["h_cluster"] = feat_util.brownclusters[self.word]
    #     else:
    #         d["h_nocluster"]=1
    #     return d

    # def i_browncluster_feats(self):
    #     d = dict()
    #     #brown cluster path feature
    #     if not feat_util.brownclusters:
    #         feat_util.init_brown()
    #     if self.word in feat_util.brownclusters:
    #         bc = feat_util.brownclusters[self.word]
    #         for i in range(1,len(bc)):
    #             d["i_cluster_"+bc[0:i] ]=1
    #
    #         #brown cluster height=general/depth=fringiness
    #         d["i_cluster_height"]=len(bc)
    #         d["i_cluster_depth"]=feat_util.cluster_heights[bc]
    #     else:
    #         #taking average
    #         #d["i_cluster_height"]=ave_brown_height
    #         #d["i_cluster_depth"]=ave_brown_depth
    #         #taking extremes
    #         d["i_nocluster"]=0
    #         d["i_cluster_height"]=0
    #         d["i_cluster_depth"]=feat_util.max_brown_depth
    #     return d

    #
    # def j_embedding_feats(self):
    #     d = dict()
    #     embeddings = feat_util.embeddings
    #     if not feat_util.embeddings:
    #         embeddings = feat_util.init_embeddings()
    #     key = self.word.lower()
    #     if key in embeddings.keys():
    #         emb=embeddings[key]
    #         for d in range(len(emb)):
    #             d["j_embed_"+str(d)] = 100*float(emb[d])
    #     else:
    #         d["j_noembed"] = 1
    #     # TODO: (1) fringiness of embedding
    #     return d

    def k_dependency_feats(self):
        wordindex = self.index + 1
        # print(self.index, self.word)
        # print(self.deptree.edges(), len(self.deptree.edges()))
        headindex = feat_util.dep_head_of(self.deptree, wordindex)
        d = dict()
        d["k_dist_to_root"] = len(
            feat_util.dep_pathtoroot(self.deptree, wordindex))
        d["k_deprel"] = self.deptree[headindex][wordindex]["deprel"]
        d["k_headdist"] = abs(headindex - wordindex)  # maybe do 0 for root?
        d["k_head_degree"] = nx.degree(self.deptree, headindex)
        d["k_child_degree"] = nx.degree(self.deptree, wordindex)
        return d

    # def l_context_feats(self):
    #     wordindex = self.index + 1
    #     headindex = feat_util.dep_head_of(self.deptree, wordindex)
    #     d = dict()
    #     d["l_brown_bag"] = "_"
    #     d["l_context_embed"] = "_"
    #     return d

    def featurize_by_type(self, feat_types):
        _d = dict()
        type2func = {'simple': self.a_simple_feats,
                     'wordnet': self.b_wordnet_feats,
                     'positional': self.c_positional_feats,
                     'frequency': self.d_frequency_feats,
                     'morphological': self.e_morphological_feats,
                     'prob_in_context': self.f_prob_in_context_feats,
                     'char_complexity': self.g_char_complexity_feats,
                     # 'brownpath': self.h_brownpath_feats,
                     # 'browncluster': self.i_browncluster_feats,
                     # 'embedding': self.j_embedding_feats,
                     'dependency': self.k_dependency_feats,
                     # 'context': self.l_context_feats
                     }
        if not feat_types:
            # use all feats if none explicitly
            feat_types = type2func.keys()
        for feat in feat_types:
            _d.update(type2func[feat]())
        return _d

    def baselinefeatures(self):
        d = dict()
        d.update(self.a_simple_feats_lexicalized())
        return d