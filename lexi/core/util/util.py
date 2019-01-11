import pickle

import numpy as np
import warnings
import pdb
import sys
import html
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

logger = logging.getLogger('lexi')


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def relative_levenshtein(s1, s2):
    """
    Levenshtein distance relative to mean word length
    :param s1:
    :param s2:
    :return:
    """
    ls = levenshtein(s1, s2)
    return ls / ((len(s1) + len(s2)) / 2)


def escape(txt):
    try:
        txt = html.escape(txt)
    except TypeError:
        sys.stderr.write("Warning! Something went wrong when escaping '{}' "
                         "(type: {})".format(txt, type(txt)))

    # txt = txt.replace('\n', '')
    return txt


def unicode(s):
    return u'{}'.format(s)


def filter_html(html):
    """
    Extracts pure text from input HTML, returns mapping from character offsets
    in pure text to HTML tags, i.e. for `<b>This is <i>pure</i> text</b>` it
    will return a dictionary {0: ["<b>"], 8: ["<i>"], ...} and a string `This is
    pure text`.
    :param html: The original HTML source
    :return: a dictionary mapping char offsets to HTML tags, and the pure text
    """
    offset2html = defaultdict(list)
    html_buffer = ""
    pure_text = ""
    in_script_tag = False
    for c in html:
        # print(c, html_buffer, pure_text)
        if c in ["<"]:
            html_buffer += c
        elif c in [">"]:
            html_buffer += c
            offset2html[len(pure_text)].append(html_buffer)
            in_script_tag = html_buffer.startswith("<script")
            html_buffer = ""
        else:
            if len(html_buffer) > 0 or in_script_tag:
                # i.e., currently within html
                html_buffer += c
            else:
                pure_text += c
    return offset2html, pure_text


def spans(txt, tokens):
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield offset, offset+len(token)
        offset += len(token)


def span_tokenize_sents(text, language="english"):
    return spans(text, sent_tokenize(text, language=language))


def span_tokenize_words(text, language="english"):
    return spans(text, word_tokenize(text, language=language))


def logzero():
    return -np.inf


def safe_log(x):
    if x == 0:
        return logzero()
    return np.log(x)


def logsum_pair(logx, logy):
    """
    Return log(x+y), avoiding arithmetic underflow/overflow.

    logx: log(x)
    logy: log(y)

    Rationale:

    x + y    = e^logx + e^logy
             = e^logx (1 + e^(logy-logx))
    log(x+y) = logx + log(1 + e^(logy-logx)) (1)

    Likewise,
    log(x+y) = logy + log(1 + e^(logx-logy)) (2)

    The computation of the exponential overflows earlier and is less precise
    for big values than for small values. Due to the presence of logy-logx
    (resp. logx-logy), (1) is preferred when logx > logy and (2) is preferred
    otherwise.
    """
    if logx == logzero():
        return logy
    elif logx > logy:
        return logx + np.log1p(np.exp(logy-logx))
    else:
        return logy + np.log1p(np.exp(logx-logy))


def logsum(logv):
    """
    Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
    """
    res = logzero()
    for val in logv:
        res = logsum_pair(res, val)
    return res


class LabelDictionary(dict):
    """This class implements a dictionary of labels. Labels as mapped to
    integers, and it is efficient to retrieve the label name from its
    integer representation, and vice-versa."""

    def __init__(self, label_names=[]):
        self.names = []
        for name in label_names:
            self.add(name)

    def add(self, name):
        label_id = len(self.names)
        if name in self:
            warnings.warn('Ignoring duplicated label ' + name)
        self[name] = label_id
        self.names.append(name)
        return label_id

    def get_label_name(self, label_id):
        return self.names[label_id]

    def get_label_id(self, name):
        return self[name]


# ----------
# Replicates the same features as the HMM
# One for word/tag and tag/tag pair
# ----------
class IDFeatures:
    """
        Base class to extract features from a particular dataset.

        feature_dic --> Dictionary of all existing features maps feature_name (string) --> feature_id (int)
        feture_names --> List of feature names. Each position is the feature_id and contains the feature name
        nr_feats --> Total number of features
        feature_list --> For each sentence in the corpus contains a pair of node feature and edge features
        dataset --> The original dataset for which the features were extracted

        Caches (for speedup):
        initial_state_feature_cache -->
        node_feature_cache -->
        edge_feature_cache -->
        final_state_feature_cache -->
    """

    def __init__(self, dataset):
        """dataset is a sequence list."""
        self.feature_dict = LabelDictionary()
        self.feature_list = []

        self.add_features = False
        self.dataset = dataset

        # Speed up
        self.node_feature_cache = {}
        self.initial_state_feature_cache = {}
        self.final_state_feature_cache = {}
        self.edge_feature_cache = {}

    def get_num_features(self):
        return len(self.feature_dict)

    def build_features(self):
        """
        Generic function to build features for a given dataset.
        Iterates through all sentences in the dataset and extracts its features,
        saving the node/edge features in feature list.
        """
        self.add_features = True
        for sequence in self.dataset.seq_list:
            initial_features, transition_features, final_features, emission_features = \
                self.get_sequence_features(sequence)
            self.feature_list.append([initial_features, transition_features, final_features, emission_features])
        self.add_features = False

    def get_sequence_features(self, sequence):
        """
        Returns the features for a given sequence.
        For a sequence of size N returns:
        Node_feature a list of size N. Each entry contains the node potentials for that position.
        Edge_features a list of size N+1.
        - Entry 0 contains the initial features
        - Entry N contains the final features
        - Entry i contains entries mapping the transition from i-1 to i.
        """
        emission_features = []
        initial_features = []
        transition_features = []
        final_features = []

        # Take care of first position
        features = []
        features = self.add_initial_features(sequence, sequence.y[0], features)
        initial_features.append(features)

        # Take care of middle positions
        for pos, tag in enumerate(sequence.y):
            features = []
            features = self.add_emission_features(sequence, pos, sequence.y[pos], features)
            emission_features.append(features)

            if pos > 0:
                prev_tag = sequence.y[pos-1]
                features = []
                features = self.add_transition_features(sequence, pos-1, tag, prev_tag, features)
                transition_features.append(features)

        # Take care of final position
        features = []
        features = self.add_final_features(sequence, sequence.y[-1], features)
        final_features.append(features)

        return initial_features, transition_features, final_features, emission_features

    # f(t,y_t,X)
    # Add the word identity and if position is
    # the first also adds the tag position
    def get_emission_features(self, sequence, pos, y):
        all_feat = []
        x = sequence.x[pos]
        if x not in self.node_feature_cache:
            self.node_feature_cache[x] = {}
        if y not in self.node_feature_cache[x]:
            node_idx = []
            node_idx = self.add_emission_features(sequence, pos, y, node_idx)
            self.node_feature_cache[x][y] = node_idx
        idx = self.node_feature_cache[x][y]
        all_feat = idx[:]
        return all_feat

    # f(t,y_t,y_(t-1),X)
    # Speed up of code
    def get_transition_features(self, sequence, pos, y, y_prev):
        assert (0 <= pos < len(sequence.x)), pdb.set_trace()

        if y not in self.edge_feature_cache:
            self.edge_feature_cache[y] = {}
        if y_prev not in self.edge_feature_cache[y]:
            edge_idx = []
            edge_idx = self.add_transition_features(sequence, pos, y, y_prev, edge_idx)
            self.edge_feature_cache[y][y_prev] = edge_idx
        return self.edge_feature_cache[y][y_prev]

    def get_initial_features(self, sequence, y):
        if y not in self.initial_state_feature_cache:
            edge_idx = []
            edge_idx = self.add_initial_features(sequence, y, edge_idx)
            self.initial_state_feature_cache[y] = edge_idx
        return self.initial_state_feature_cache[y]

    def get_final_features(self, sequence, y_prev):
        if y_prev not in self.final_state_feature_cache:
            edge_idx = []
            edge_idx = self.add_final_features(sequence, y_prev, edge_idx)
            self.final_state_feature_cache[y_prev] = edge_idx
        return self.final_state_feature_cache[y_prev]

    def add_initial_features(self, sequence, y, features):
        # Get label name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Generate feature name.
        feat_name = "init_tag:%s" % y_name
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)
        return features

    def add_final_features(self, sequence, y_prev, features):
        # Get label name from ID.
        y_name = self.dataset.y_dict.get_label_name(y_prev)
        # Generate feature name.
        feat_name = "final_prev_tag:%s" % y_name
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)
        return features

    def add_emission_features(self, sequence, pos, y, features):
        """Add word-tag pair feature."""
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get word name from ID.
        x_name = self.dataset.x_dict.get_label_name(x)
        # Generate feature name.
        feat_name = "id:%s::%s" % (x_name, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)
        return features

    def add_transition_features(self, sequence, pos, y, y_prev, features):
        """ Adds a feature to the edge feature list.
        Creates a unique id if its the first time the feature is visited
        or returns the existing id otherwise
        """
        assert pos < len(sequence.x)-1, pdb.set_trace()

        # Get label name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get previous label name from ID.
        y_prev_name = self.dataset.y_dict.get_label_name(y_prev)
        # Generate feature name.
        feat_name = "prev_tag:%s::%s" % (y_prev_name, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)
        return features

    def add_feature(self, feat_name):
        """
        Builds a dictionary of feature name to feature id
        If we are at test time and we don't have the feature
        we return -1.
        """
        # Check if feature exists and if so, return the feature ID.
        if feat_name in self.feature_dict:
            return self.feature_dict[feat_name]
        # If 'add_features' is True, add the feature to the feature
        # dictionary and return the feature ID. Otherwise return -1.
        if not self.add_features:
            return -1
        return self.feature_dict.add(feat_name)


# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get word name from ID.
        x_name = self.dataset.x_dict.get_label_name(x)
        word = unicode(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        if word.istitle():
            # Generate feature name.
            feat_name = "uppercased::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if word.isdigit():
            # Generate feature name.
            feat_name = "number::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if word.find("-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in range(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # Prefixes
        max_prefix = 3
        for i in range(max_prefix):
            if len(word) > i+1:
                prefix = word[:i+1]
                # Generate feature name.
                feat_name = "prefix:%s::%s" % (prefix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        return features


class SequenceClassificationDecoder:
    """ Implements a sequence classification decoder."""
    def __init__(self):
        pass

    # ----------
    # Computes the forward trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_forward(self, initial_scores, transition_scores, final_scores,
                    emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Forward variables.
        forward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        forward[0, :] = emission_scores[0, :] + initial_scores

        # Forward loop.
        for pos in range(1, length):
            for current_state in range(num_states):
                # Note the fact that multiplication in log domain turns a sum and sum turns a logsum
                forward[pos, current_state] = logsum(
                    forward[pos - 1, :] + transition_scores[pos - 1,
                                          current_state, :])
                forward[pos, current_state] += emission_scores[
                    pos, current_state]

        # Termination.
        log_likelihood = logsum(forward[length - 1, :] + final_scores)

        return log_likelihood, forward

    # ----------
    # Computes the backward trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_backward(self, initial_scores, transition_scores, final_scores,
                     emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Backward variables.
        backward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        backward[length - 1, :] = final_scores

        # Backward loop.
        for pos in range(length - 2, -1, -1):
            for current_state in range(num_states):
                backward[pos, current_state] = \
                    logsum(backward[pos + 1, :] +
                           transition_scores[pos, :, current_state] +
                           emission_scores[pos + 1, :])

        # Termination.
        log_likelihood = logsum(
            backward[0, :] + initial_scores + emission_scores[0, :])

        return log_likelihood, backward

    # ----------
    # Computes the viterbi trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    # ----------
    def run_viterbi(self, initial_scores, transition_scores, final_scores,
                    emission_scores):

        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Variables storing the Viterbi scores.
        viterbi = np.zeros([length, num_states]) + logzero()

        # Variables storing the paths to backtrack.
        backtrack = -np.ones([length, num_states], dtype=int)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Complete Exercise 2.8
        # raise NotImplementedError("Complete Exercise 2.8")

        #### Little guide of the implementation ####################################
        # Initializatize the viterbi scores
        #
        # Do the double of the viterbi loop (lines 7 to 12 in the guide pdf)
        # from 1 to length
        #     from 0 to num_states
        #       ...
        #
        # define the best_path and best_score
        #
        # backtrack the best_path using the viterbi paths (lines 17-18 pseudocode in the guide pdf)
        #
        # return best_path and best_score
        ############################################################################

        for ck in range(num_states):
            viterbi[0, :] = initial_scores + emission_scores[0, :]

        for i in range(1, length):
            for ck in range(num_states):
                cells = transition_scores[i - 1, ck, :] + viterbi[i - 1, :]
                viterbi[i, ck] = np.max(cells) + emission_scores[i, ck]
                backtrack[i, ck] = np.argmax(cells)

        # lastcells = []
        # for ck in range(num_states):
        #    lastcells.append(np.max([final_scores[cl] * viterbi[-1, cl] for cl in range(num_states)]))
        lastcells = final_scores + viterbi[-1, :]

        best_score = np.max(lastcells)
        # best_path = []
        # best_path.append(y_N)
        best_path[-1] = np.argmax(lastcells)
        for i in range(length - 2, -1, -1):
            best_path[i] = backtrack[i + 1, best_path[i + 1]]
        return best_path, best_score

    def run_forward_backward(self, initial_scores, transition_scores,
                             final_scores, emission_scores):
        log_likelihood, forward = self.run_forward(initial_scores,
                                                   transition_scores,
                                                   final_scores,
                                                   emission_scores)
        print('Log-Likelihood =', log_likelihood)

        log_likelihood, backward = self.run_backward(initial_scores,
                                                     transition_scores,
                                                     final_scores,
                                                     emission_scores)
        print('Log-Likelihood =', log_likelihood)
        return forward, backward

#
#
# class PostagCorpus(object):
#
#     def __init__(self):
#         # Word dictionary.
#         self.word_dict = LabelDictionary()
#
#         # POS tag dictionary.
#         # Initialize noun to be tag zero so that it the default tag.
#         self.tag_dict = LabelDictionary(['noun'])
#
#         # Initialize sequence list.
#         self.sequence_list = SequenceList(self.word_dict, self.tag_dict)
#
#     # Read a text file in conll format and return a sequence list
#     #
#     def read_sequence_list_conll(self, train_file,
#                                  mapping_file=("%s/en-ptb.map"
#                                                % dirname(__file__)),
#                                  max_sent_len=100000,
#                                  max_nr_sent=100000):
#
#         # Build mapping of postags:
#         mapping = {}
#         if mapping_file is not None:
#             for line in open(mapping_file):
#                 coarse, fine = line.strip().split("\t")
#                 mapping[coarse.lower()] = fine.lower()
#         instance_list = self.read_conll_instances(train_file,
#                                                   max_sent_len,
#                                                   max_nr_sent, mapping)
#         seq_list = SequenceList(self.word_dict, self.tag_dict)
#         for sent_x, sent_y in instance_list:
#             seq_list.add_sequence(sent_x, sent_y)
#
#         return seq_list
#
#     # ----------
#     # Reads a conll file into a sequence list.
#     # ----------
#     def read_conll_instances(self, file, max_sent_len, max_nr_sent, mapping):
#         if file.endswith("gz"):
#             zf = gzip.open(file, 'rb')
#             reader = codecs.getreader("utf-8")
#             contents = reader(zf)
#         else:
#             contents = codecs.open(file, "r", "utf-8")
#
#         nr_sent = 0
#         instances = []
#         ex_x = []
#         ex_y = []
#         nr_types = len(self.word_dict)
#         nr_pos = len(self.tag_dict)
#         for line in contents:
#             toks = line.split()
#             if len(toks) < 2:
#                 # print "sent n %i size %i"%(nr_sent,len(ex_x))
#                 if len(ex_x) < max_sent_len and len(ex_x) > 1:
#                     # print "accept"
#                     nr_sent += 1
#                     instances.append([ex_x, ex_y])
#                 # else:
#                 #     if(len(ex_x) <= 1):
#                 #         print "refusing sentence of len 1"
#                 if nr_sent >= max_nr_sent:
#                     break
#                 ex_x = []
#                 ex_y = []
#             else:
#                 pos = toks[4]
#                 word = toks[1]
#                 pos = pos.lower()
#                 if pos not in mapping:
#                     mapping[pos] = "noun"
#                     print "unknown tag %s" % pos
#                 pos = mapping[pos]
#                 if word not in self.word_dict:
#                     self.word_dict.add(word)
#                 if pos not in self.tag_dict:
#                     self.tag_dict.add(pos)
#                 ex_x.append(word)
#                 ex_y.append(pos)
#                 # ex_x.append(self.word_dict[word])
#                 # ex_y.append(self.tag_dict[pos])
#         return instances
#
#     # Read a text file in brown format and return a sequence list
#     #
#     # def read_sequence_list_brown(self,mapping_file="readers/en-ptb.map",max_sent_len=100000,max_nr_sent=100000,categories=""):
#     #     ##Build mapping of postags:
#     #     mapping = {}
#     #     if(mapping_file != None):
#     #         for line in open(mapping_file):
#     #             coarse,fine = line.strip().split("\t")
#     #             mapping[coarse.lower()] = fine.lower()
#
#     #     if(categories == ""):
#     #         sents = brown.tagged_sents()
#     #     else:
#     #         sents = brown.tagged_sents(categories=categories)
#     #     seq_list = Sequence_List(self.word_dict,self.int_to_word,self.tag_dict,self.int_to_tag)
#     #     nr_types = len(self.word_dict)
#     #     nr_tag = len(self.tag_dict)
#     #     for sent in sents:
#     #         if(len(sent) > max_sent_len or len(sent) <= 1):
#     #             continue
#     #         ns_x = []
#     #         ns_y = []
#     #         for word,tag in sent:
#     #                 tag = tag.lower()
#     #                 if(tag not in mapping):
#     #                     ##Add unk tags to dict
#     #                     mapping[tag] = "noun"
#     #                 c_t =  mapping[tag]
#     #                 if(word not in self.word_dict):
#     #                     self.word_dict[word] = nr_types
#     #                     c_word = nr_types
#     #                     self.int_to_word.append(word)
#     #                     nr_types += 1
#     #                 else:
#     #                     c_word = self.word_dict[word]
#     #                 if(c_t not in self.tag_dict):
#     #                     self.tag_dict[c_t] = nr_tag
#     #                     c_pos_c = nr_tag
#     #                     self.int_to_tag.append(c_t)
#     #                     nr_tag += 1
#     #                 else:
#     #                     c_pos_c = self.tag_dict[c_t]
#     #                 ns_x.append(c_word)
#     #                 ns_y.append(c_pos_c)
#     #         seq_list.add_sequence(ns_x,ns_y)
#     #     return seq_list
#
#     # Dumps a corpus into a file
#     def save_corpus(self, dir):
#         if not os.path.isdir(dir + "/"):
#             os.mkdir(dir + "/")
#         word_fn = codecs.open(dir + "word.dic", "w", "utf-8")
#         for word_id, word in enumerate(self.int_to_word):
#             word_fn.write("%i\t%s\n" % (word_id, word))
#         word_fn.close()
#         tag_fn = open(dir + "tag.dic", "w")
#         for tag_id, tag in enumerate(self.int_to_tag):
#             tag_fn.write("%i\t%s\n" % (tag_id, tag))
#         tag_fn.close()
#         word_count_fn = open(dir + "word.count", "w")
#         for word_id, counts in self.word_counts.iteritems():
#             word_count_fn.write("%i\t%s\n" % (word_id, counts))
#         word_count_fn.close()
#         self.sequence_list.save(dir + "sequence_list")
#
#     # Loads a corpus from a file
#     def load_corpus(self, dir):
#         word_fn = codecs.open(dir + "word.dic", "r", "utf-8")
#         for line in word_fn:
#             word_nr, word = line.strip().split("\t")
#             self.int_to_word.append(word)
#             self.word_dict[word] = int(word_nr)
#         word_fn.close()
#         tag_fn = open(dir + "tag.dic", "r")
#         for line in tag_fn:
#             tag_nr, tag = line.strip().split("\t")
#             if tag not in self.tag_dict:
#                 self.int_to_tag.append(tag)
#                 self.tag_dict[tag] = int(tag_nr)
#         tag_fn.close()
#         word_count_fn = open(dir + "word.count", "r")
#         for line in word_count_fn:
#             word_nr, word_count = line.strip().split("\t")
#             self.word_counts[int(word_nr)] = int(word_count)
#         word_count_fn.close()
#         self.sequence_list.load(dir + "sequence_list")
# detokenizer = MosesDetokenizer()

