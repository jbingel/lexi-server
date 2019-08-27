import gensim
import numpy as np


class W2VModelEnsemble:

    def __init__(self, models):
        self.models = models

    def most_similar(self, target, min_similarity=0.5, topn=10):

        all_similar_words = set()
        for model in self.models:
            if target in model:
                all_similar_words.update([w for w, sim in
                                          model.most_similar(target, topn=topn)
                                          if sim > min_similarity])
        candidate_mean_scores = []
        for w in all_similar_words:
            mean_score = np.mean([model.similarity(target, w)
                                  for model in self.models
                                  if w in model and target in model])
            candidate_mean_scores.append((w, mean_score))

        # sort
        most_similar = sorted(candidate_mean_scores, key=lambda x: x[1],
                              reverse=True)
        # select top n
        return most_similar[:topn]

    def similarity(self, w1, w2):
        return np.mean([model.similarity(w1, w2) for model in self.models])


def make_synonyms_dict(synonyms_file):
    """

    :param synonyms_file:
    :return:
    """
    from collections import defaultdict
    words2synonyms = defaultdict(set)
    for line in open(synonyms_file, encoding='utf8'):
        tgt, syns = line.strip().split("\t", 1)
        words2synonyms[tgt].update(syns.split(";"))
    return words2synonyms


def parse_embeddings(embeddings_files):
    individual_models = []
    for model_file in embeddings_files:
        try:
            _model = gensim.models.KeyedVectors.load_word2vec_format(
                model_file, binary=True, unicode_errors='ignore')
        except UnicodeDecodeError:
            try:
                _model = gensim.models.KeyedVectors.load(model_file)
            except:
                continue
        individual_models.append(_model)
    return W2VModelEnsemble(individual_models)
