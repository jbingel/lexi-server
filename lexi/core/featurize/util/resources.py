import gzip
import os
import pickle

from lexi.core.featurize.util import lm

etymwn = None
lm_reg = None
brownclusters = None
cluster_heights = None
ave_brown_depth = None
ave_brown_height = None
max_brown_depth = None
embeddings = None


def init_etymwn():
    global etymwn
    etymwn=loadEtym()
    return etymwn


def init_lms():
    global lm_reg
    print('Reading language models... ')
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    lm_words_swp = read_lm(scriptdir+"/../../data/lms/simplewiki.arpa")
    lm_words_wp = read_lm(scriptdir+"/../../data/lms/enwiki.arpa")
    lm_chars_swp = read_lm(scriptdir+"/../../data/lms/simplewiki_chars.arpa")
    lm_chars_wp = read_lm(scriptdir+"/../../data/lms/enwiki_chars.arpa")
    lm_reg = {"words": {"swp": lm_words_swp, "wp": lm_words_wp},
              "chars": {"swp": lm_chars_swp, "wp": lm_chars_wp}}
    return lm_reg


def init_brown():
    global brownclusters, cluster_heights, ave_brown_depth, ave_brown_height, max_brown_depth
    brownclusters, cluster_heights, ave_brown_depth, ave_brown_height, max_brown_depth = \
        read_brown_clusters('/coastal/brown_clusters/rcv1.64M-c1000-p1.paths', 1000)


def init_embeddings():
    global embeddings
    embeddings = read_embeddings('/home/joachim/data/mono_embeddings/glove.6B.300d.txt.gz')
    return embeddings


def read_brown_clusters(src, total_clusters):
    print('\tReading brown clusters...')
    d = {}
    infile = open(src, 'r')
    c = []
    ch = {}
    total_words = 0.0
    total_depths = 0.0
    max_depth = 0
    for line in infile.readlines():
        data = line.strip().split('\t')
        d[data[1]] = data[0]
        total_words += 1
        total_depths += len(data[0])
        if len(data[0]) > max_depth:
            max_depth = len(data[0])
        if data[0] not in c:
            c.append(data[0])
    print('Done!')
    # calculation of heights
    print('  Calculating brown heights...')
    total_heights = 0.0
    c.sort(key=len, reverse=True)
    for x in c:
        if x[:-1] not in ch.keys():
            ch[x] = 1
        else:
            ch[x] = ch[x[:-1]]+1
    for word in d.keys():
        total_heights = total_heights+ch[d[word]]
    print('Done!')
    return d, ch, total_depths/total_words, total_heights/total_words, max_depth


def read_embeddings(src):
    print('Reading embeddings...')
    d = {}
    infile = gzip.open(src, 'rt')
    for line in infile.readlines():
        data = line.strip().split(' ')
        d[data[0]] = data[1:]
    print('Done!')
    return d


def read_lm(src):
    return lm.LM(src)


def loadEtym():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    etym_file = scriptdir+"/../../data/other/etymwn.pickle"
    with open(etym_file, 'rb') as pickle_file:
        G = pickle.load(pickle_file)
    return G