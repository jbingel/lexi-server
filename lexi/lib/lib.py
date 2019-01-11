import logging
import pickle

import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from collections import defaultdict

from lexi.config import RANKER_MODEL_PATH_TEMPLATE

logger = logging.getLogger('lexi')


def make_synonyms_dict(synonyms_file):
    """

    :param synonyms_file:
    :return:
    """
    words2synonyms = defaultdict(set)
    for line in open(synonyms_file):
        tgt, syns = line.strip().split("\t", 1)
        words2synonyms[tgt].update(syns.split(";"))
    return words2synonyms


class Generator:
    def __init__(self):
        raise NotImplementedError

    def getSubstitutionsSingle(self, sentence, target, index, **kwargs):
        raise NotImplementedError


class SynonymDBGenerator(Generator):
    """
    Generates candidates from a serialized WordNet-like list of synonymy
    relations.
    """

    def __init__(self, synonyms_file):
        self.word2synonmys = make_synonyms_dict(synonyms_file)

    def getSubstitutionsSingle(self, sentence, target, index, **kwargs):
        # TODO get POS of word for filtering?
        """

        :param sentence:
        :param target:
        :param index:
        :return:
        """
        return {target: self.word2synonmys.get(target, {})}


class LexensteinGenerator(Generator):

    def __init__(self, w2vmodels):
        import gensim
        self.model = None
        self.individual_models = []
        for model_file in w2vmodels:
            try:
                _model = gensim.models.KeyedVectors.load_word2vec_format(
                    model_file, binary=True, unicode_errors='ignore')
            except UnicodeDecodeError:
                try:
                    _model = gensim.models.KeyedVectors.load(model_file)
                except:
                    continue
            self.individual_models.append(_model)
        logger.debug(self.individual_models)
        self.model = W2VModelEnsemble(self.individual_models)

    def getSubstitutionsSingle(self, sentence, target, index,
                               min_similarity=0.2):
        """
        :param sentence:
        :param target:
        :param index:
        :param min_similarity: minimum similarity score
        :return:
        """
        if min_similarity <= 0 or min_similarity > 1:
            raise ValueError("'min_similarity' must be between 0 and 1 "
                             "(you provided {}).".format(min_similarity))
        substitutions = self.getInitialSet([[sentence, target, index]],
                                           min_similarity)
        return substitutions

    def getInitialSet(self, data, min_similarity):
        trgs = []
        for i in range(len(data)):
            d = data[i]
            logger.debug(d)
            target = d[1].strip().lower()
            head = int(d[2].strip())
            trgs.append(target)

        logger.debug("tgts: {}".format(trgs))
        logger.debug("  getting candidates with min_similarity={}".
                     format(min_similarity))
        subs = []
        cands = set([])
        for i in range(len(data)):
            d = data[i]
            t = trgs[i]

            word = t

            most_sim = self.model.most_similar(word)

            subs.append([word for word, score in most_sim
                         if score >= min_similarity])

        logger.debug("subs: {}".format(subs))
        subsr = subs
        subs = []
        for l in subsr:
            lr = []
            for inst in l:
                cand = inst.split('|||')[0].strip()
                cands.add(cand)
                lr.append(inst)
            subs.append(lr)

        cands = list(cands)

        subs_filtered = self.filterSubs(data, subs, trgs)

        final_cands = {}
        for i in range(0, len(data)):
            target = data[i][1]
            logger.debug(subs_filtered)
            cands = subs_filtered[i][0:len(subs_filtered[i])]
            cands = [word.split('|||')[0].strip() for word in cands]
            if target not in final_cands:
                final_cands[target] = set([])
            final_cands[target].update(set(cands))

        return final_cands

    def filterSubs(self, data, subs, trgs):
        result = []
        for i in range(0, len(data)):
            d = data[i]

            t = trgs[i]

            most_sim = subs[i]
            most_simf = []

            for cand in most_sim:
                if cand!=t:
                    most_simf.append(cand)

            result.append(most_simf)
        return result


class EnsembleLexensteinGenerator(LexensteinGenerator):

    def __init__(self, w2vmodels):
        import gensim
        self.model = None
        self.individual_models = []
        for model_file in w2vmodels:
            try:
                _model = gensim.models.KeyedVectors.load_word2vec_format(
                    model_file, binary=True, unicode_errors='ignore')
            except UnicodeDecodeError:
                try:
                    _model = gensim.models.KeyedVectors.load(model_file)
                except:
                    continue
            self.individual_models.append(_model)
        self.model = W2VModelEnsemble(self.individual_models)

    def getInitialSet(self, data, amount=5, min_similarity=0.5):

        trgs = []
        for i in range(len(data)):
            d = data[i]
            logger.debug(d)
            target = d[1].strip().lower()
            head = int(d[2].strip())
            trgs.append(target)

        logger.debug("tgts: {}".format(trgs))
        subs = []
        cands = set([])
        candidates = set()
        for i in range(len(data)):
            d = data[i]
            t = trgs[i]
            for model in self.models:
                try:
                    candidates.update([(w, v) for w, v in
                                 model.most_similar(t.decode('utf-8'), topn=10)
                                 if v > min_similarity])
                except Exception:
                    try:
                        candidates.update([(w, v) for w, v in
                                           model.most_similar(t, topn=10)
                                      if v > min_similarity])
                    except Exception:
                        pass

            candidate_mean_scores = []
            for candidate in candidates:
                # compute mean score for every candidate across models
                mean_score = np.mean([model.similarity(t, candidate)
                                      for model in self.models
                                      if candidate in model])
                candidate_mean_scores.append((candidate, mean_score))

            # sort candidates by score (best first)
            candidate_mean_scores = sorted(candidate_mean_scores,
                                           key=lambda x: x[1], reversed=True)
            # select top n
            best_candidates = [cand for cand, sim in
                               candidate_mean_scores][:amount]
            # subs.append([word[0] for word in most_sim])
            subs.append(best_candidates)

        logger.debug("tgts: {}".format(trgs))
        subsr = subs
        subs = []
        for l in subsr:
            lr = []
            for inst in l:
                cand = inst.split('|||')[0].strip()
                cands.add(cand)
                lr.append(inst)
            subs.append(lr)

        cands = list(cands)

        subs_filtered = self.filterSubs(data, subs, trgs)

        final_cands = {}
        for i in range(0, len(data)):
            target = data[i][1]
            logger.debug(subs_filtered, amount, i)
            cands = subs_filtered[i][0:min(amount, len(subs_filtered[i]))]
            cands = [word.split('|||')[0].strip() for word in cands]
            if target not in final_cands:
                final_cands[target] = set([])
            final_cands[target].update(set(cands))

        return final_cands

    def filterSubs(self, data, subs, trgs):
        result = []
        for i in range(0, len(data)):
            d = data[i]

            t = trgs[i]

            most_sim = subs[i]
            most_simf = []

            for cand in most_sim:
                if cand!=t:
                    most_simf.append(cand)

            result.append(most_simf)
        return result


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


class BoundaryRanker:

    def __init__(self, fe=None, userId=None):
        self.fe = fe
        self.classifier = None
        self.feature_selector = None
        self.userId = userId

    def trainRankerWithCrossValidation(
            self, victor_corpus, positive_range, folds, test_size,
            losses=['hinge', 'modified_huber'], penalties=['elasticnet'],
            alphas=[0.0001, 0.001, 0.01],
            l1_ratios=[0.0, 0.15, 0.25, 0.5, 0.75, 1.0], k='all'):
        # Read victor corpus:
        data = []
        f = open(victor_corpus)
        for line in f:
            data.append(line.strip().split('\t'))
        f.close()

        # Create matrixes:
        X = self.fe.calculateFeatures(victor_corpus)
        Y = self.generateLabels(data, positive_range)

        # Select features:
        self.feature_selector = SelectKBest(f_classif, k=k)
        self.feature_selector.fit(X, Y)
        X = self.feature_selector.transform(X)

        # Extract ranking problems:
        firsts = []
        candidates = []
        Xsets = []
        Ysets = []
        index = -1
        for line in data:
            fs = set([])
            cs = []
            Xs = []
            Ys = []
            for cand in line[3:len(line)]:
                index += 1
                candd = cand.split(':')
                rank = candd[0].strip()
                word = candd[1].strip()

                cs.append(word)
                Xs.append(X[index])
                Ys.append(Y[index])
                if rank=='1':
                    fs.add(word)
            firsts.append(fs)
            candidates.append(cs)
            Xsets.append(Xs)
            Ysets.append(Ys)

        # Create data splits:
        datasets = []
        for i in range(0, folds):
            Xtr, Xte, Ytr, Yte, Ftr, Fte, Ctr, Cte = train_test_split(
                Xsets, Ysets, firsts, candidates, test_size=test_size,
                random_state=i)
            Xtra = []
            for matrix in Xtr:
                Xtra += matrix
            Xtea = []
            for matrix in Xte:
                Xtea += matrix
            Ytra = []
            for matrix in Ytr:
                Ytra += matrix
            datasets.append((Xtra, Ytra, Xte, Xtea, Fte, Cte))

        # Get classifier with best parameters:
        max_score = -1.0
        parameters = ()
        for l in losses:
            for p in penalties:
                for a in alphas:
                    for r in l1_ratios:
                        sum = 0.0
                        sum_total = 0
                        for dataset in datasets:
                            Xtra = dataset[0]
                            Ytra = dataset[1]
                            Xte = dataset[2]
                            Xtea = dataset[3]
                            Fte = dataset[4]
                            Cte = dataset[5]

                            classifier = linear_model.SGDClassifier(loss=l, penalty=p, alpha=a, l1_ratio=r, epsilon=0.0001)
                            try:
                                classifier.fit(Xtra, Ytra)
                                t1 = self.getCrossValidationScore(classifier, Xtea, Xte, Fte, Cte)
                                sum += t1
                                sum_total += 1
                            except Exception:
                                pass
                        sum_total = max(1, sum_total)
                        if (sum/sum_total)>max_score:
                            max_score = sum
                            parameters = (l, p, a, r)
        self.classifier = linear_model.SGDClassifier(loss=parameters[0], penalty=parameters[1], alpha=parameters[2], l1_ratio=parameters[3], epsilon=0.0001)
        self.classifier.fit(X, Y)

    def getCrossValidationScore(self, classifier, Xtea, Xte, firsts, candidates):
        distances = classifier.decision_function(Xtea)
        index = -1
        corrects = 0
        total = 0
        for i in range(0, len(Xte)):
            xset = Xte[i]
            maxd = -999999
            for j in range(0, len(xset)):
                index += 1
                distance = distances[index]
                if distance>maxd:
                    maxd = distance
                    maxc = candidates[i][j]
            if maxc in firsts[i]:
                corrects += 1
            total += 1
        return float(corrects)/float(total)

    def getRankings(self, data):
        #Transform data:
        textdata = ''
        for inst in data:
            for token in inst:
                textdata += token+'\t'
            textdata += '\n'
        textdata = textdata.strip()

        #Create matrixes:
        X = self.fe.calculateFeatures(textdata, input='text')

        #Select features:
        X = self.feature_selector.transform(X)

        #Get boundary distances:
        distances = self.classifier.decision_function(X)

        #Get rankings:
        result = []
        index = 0
        for i in range(0, len(data)):
            line = data[i]
            scores = {}
            for subst in line[3:len(line)]:
                word = subst.strip().split(':')[1].strip()
                scores[word] = distances[index]
                index += 1
            ranking_data = sorted(list(scores.keys()), key=scores.__getitem__, reverse=True)
            result.append(ranking_data)

        #Return rankings:
        return result

    def generateLabels(self, data, positive_range):
        Y = []
        for line in data:
            max_range = min(int(line[len(line)-1].split(':')[0].strip()), positive_range)
            for i in range(3, len(line)):
                rank_index = int(line[i].split(':')[0].strip())
                if rank_index<=max_range:
                    Y.append(1)
                else:
                    Y.append(0)
        return Y

    def save(self, userId):
        with open(RANKER_MODEL_PATH_TEMPLATE.format(userId), 'wb') as pf:
            pickle.dump((self.fe, self.classifier, self.feature_selector), pf,
                        pickle.HIGHEST_PROTOCOL)

    def load(self, userId=None):
        if not userId:
            userId = self.userId
        with open(RANKER_MODEL_PATH_TEMPLATE.format(userId), 'rb') as pf:
            (self.fe, self.classifier, self.feature_selector) = pickle.load(pf)
        return self


class BoundarySelector:

    def __init__(self, boundary_ranker):
        self.ranker = boundary_ranker

    def trainSelectorWithCrossValidation(self, victor_corpus, positive_range,
                                         folds, test_size,
                                         losses=['hinge', 'modified_huber'],
                                         penalties=['elasticnet'],
                                         alphas=[0.0001, 0.001, 0.01],
                                         l1_ratios=[0.0, 0.15, 0.25, 0.5, 0.75, 1.0],
                                         k='all'):
        self.ranker.trainRankerWithCrossValidation(victor_corpus, positive_range,
                                                   folds, test_size, losses=losses, penalties=penalties, alphas=alphas, l1_ratios=l1_ratios, k=k)

    def selectCandidates(self, data, proportion, proportion_type='percentage'):
        rankings = self.ranker.getRankings(data)
        logger.debug((data, rankings))
        selected_substitutions = []

        index = -1
        for line in data:
            index += 1

            if proportion_type == 'percentage':
                if proportion > 1.0:
                    select_n = len(rankings[index])
                else:
                    select_n = int(float(proportion) * len(rankings[index]))
                selected_candidates = rankings[index][:max(1, select_n)]
            else:
                if proportion < 1:
                    toselect = 1
                elif proportion > len(rankings[index]):
                    toselect = len(rankings[index])
                else:
                    toselect = proportion
                selected_candidates = rankings[index][:toselect]

            selected_substitutions.append(selected_candidates)

        return selected_substitutions


class GlavasRanker:

    def __init__(self, fe):
        """
        Creates an instance of the GlavasRanker class.

        @param fe: A configured FeatureEstimator object.
        """

        self.fe = fe
        self.feature_values = None

    def getRankings(self, alldata):

        #Calculate features:
        textdata = ''
        for inst in alldata:
                for token in inst:
                        textdata += token+'\t'
                textdata += '\n'
        textdata = textdata.strip()
        self.feature_values = self.fe.calculateFeatures(textdata, input='text')

        #Create object for results:
        result = []

        #Read feature values for each candidate in victor corpus:
        index = 0
        for data in alldata:
            #Get all substitutions in ranking instance:
            substitutions = data[3:len(data)]

            #Get instance's feature values:
            instance_features = []
            for substitution in substitutions:
                instance_features.append(self.feature_values[index])
                index += 1

            rankings = {}
            for i in range(0, len(self.fe.identifiers)):
                #Create dictionary of substitution to feature value:
                scores = {}
                for j in range(0, len(substitutions)):
                    substitution = substitutions[j]
                    word = substitution.strip().split(':')[1].strip()
                    scores[word] = instance_features[j][i]

                #Check if feature is simplicity or complexity measure:
                rev = False
                if self.fe.identifiers[i][1]=='Simplicity':
                    rev = True

                #Sort substitutions:
                words = list(scores.keys())
                sorted_substitutions = sorted(words, key=scores.__getitem__, reverse=rev)

                #Update rankings:
                for j in range(0, len(sorted_substitutions)):
                    word = sorted_substitutions[j]
                    if word in rankings:
                        rankings[word] += j
                    else:
                        rankings[word] = j

            #Produce final rankings:
            final_rankings = sorted(list(rankings.keys()), key=rankings.__getitem__)

            #Add them to result:
            result.append(final_rankings)

        #Return result:
        return result


class NNRegressionRanker:

    def __init__(self, fe, model):
        self.fe = fe
        self.model = model

    def getRankings(self, data):
        #Transform data:
        textdata = ''
        for inst in data:
            for token in inst:
                textdata += token+'\t'
            textdata += '\n'
        textdata = textdata.strip()

        #Create matrix:
        features = self.fe.calculateFeatures(textdata, input='text')

        ranks = []
        c = -1
        for line in data:
            cands = [cand.strip().split(':')[1].strip() for cand in line[3:]]
            featmap = {}
            scoremap = {}
            for cand in cands:
                c += 1
                featmap[cand] = features[c]
                scoremap[cand] = 0.0
            for i in range(0, len(cands)-1):
                cand1 = cands[i]
                for j in range(i+1, len(cands)):
                    cand2 = cands[j]
                    posneg = np.concatenate((featmap[cand1], featmap[cand2]))
                    probs = self.model.predict(np.array([posneg]))
                    score = probs[0]
                    scoremap[cand1] += score
                    negpos = np.concatenate((featmap[cand2], featmap[cand1]))
                    probs = self.model.predict(np.array([negpos]))
                    score = probs[0]
                    scoremap[cand1] -= score
            rank = sorted(list(scoremap.keys()), key=scoremap.__getitem__, reverse=True)
            if len(rank)>1:
                if rank[0]==line[1].strip():
                    rank = rank[1:]
            ranks.append(rank)
        return ranks


class OnlineRegressionRanker:

    def __init__(self, fe, model, training_dataset=None, userId=None):
        self.fe = fe
        self.userId = userId
        if model:
            self.model = model
        elif training_dataset:
            self.model = self.trainRegressionModel(training_dataset)
        else:
            self.model = None

    def trainRegressionModel(self, training_dataset):
        # Create matrix:
        features = self.fe.calculateFeatures(training_dataset, input='file')
        Xtr = []
        Ytr = []
        f = open(training_dataset)
        c = -1
        for line in f:
            data = line.strip().split('\t')
            cands = [cand.strip().split(':')[1] for cand in data[3:]]
            indexes = [int(cand.strip().split(':')[0]) for cand in data[3:]]
            featmap = {}
            for cand in cands:
                c += 1
                featmap[cand] = features[c]
            for i in range(0, len(cands)-1):
                for j in range(i+1, len(cands)):
                    indexi = indexes[i]
                    indexj = indexes[j]
                    indexdiffji = indexj-indexi
                    indexdiffij = indexi-indexj
                    positive = featmap[cands[i]]
                    negative = featmap[cands[j]]
                    v1 = np.concatenate((positive,negative))
                    v2 = np.concatenate((negative,positive))
                    Xtr.append(v1)
                    Xtr.append(v2)
                    Ytr.append(indexdiffji)
                    Ytr.append(indexdiffij)
        f.close()
        Xtr = np.array(Xtr)
        Ytr = np.array(Ytr)

        model = linear_model.SGDRegressor()
        model.fit(Xtr, Ytr)
        return model

    def onlineTrainRegressionModel(self, training_data_text):
        logger.info("Partially fitting the ranker")
        # Create matrix:
        features = self.fe.calculateFeatures(training_data_text,
                                             format='victor', input='text')
        Xtr = []
        Ytr = []
        c = -1
        for line in training_data_text.strip().split('\n'):
            logger.debug(line)
            data = line.strip().split('\t')
            cands = [cand.strip().split(':')[1] for cand in data[3:]]
            indexes = [int(cand.strip().split(':')[0]) for cand in data[3:]]
            featmap = {}
            for cand in cands:
                c += 1
                featmap[cand] = features[c]
            for i in range(0, len(cands) - 1):
                for j in range(i + 1, len(cands)):
                    indexi = indexes[i]
                    indexj = indexes[j]
                    indexdiffji = indexj - indexi
                    indexdiffij = indexi - indexj
                    positive = featmap[cands[i]]
                    negative = featmap[cands[j]]
                    v1 = np.concatenate((positive, negative))
                    v2 = np.concatenate((negative, positive))
                    Xtr.append(v1)
                    Xtr.append(v2)
                    Ytr.append(indexdiffji)
                    Ytr.append(indexdiffij)
        Xtr = np.array(Xtr)
        Ytr = np.array(Ytr)

        self.model.partial_fit(Xtr, Ytr)
        return self.model

    def getRankings(self, data):
        #Transform data:
        textdata = ''
        for inst in data:
            for token in inst:
                textdata += token+'\t'
            textdata += '\n'
        textdata = textdata.strip()

        #Create matrix:
        features = self.fe.calculateFeatures(textdata, input='text')

        ranks = []
        c = -1
        for line in data:
            cands = [cand.strip().split(':')[1].strip() for cand in line[3:]]
            featmap = {}
            scoremap = {}
            for cand in cands:
                c += 1
                featmap[cand] = features[c]
                scoremap[cand] = 0.0
            for i in range(0, len(cands)-1):
                cand1 = cands[i]
                for j in range(i+1, len(cands)):
                    cand2 = cands[j]
                    posneg = np.concatenate((featmap[cand1], featmap[cand2]))
                    probs = self.model.predict(np.array([posneg]))
                    score = probs[0]
                    scoremap[cand1] += score
                    negpos = np.concatenate((featmap[cand2], featmap[cand1]))
                    probs = self.model.predict(np.array([negpos]))
                    score = probs[0]
                    scoremap[cand1] -= score
            rank = sorted(list(scoremap.keys()), key=scoremap.__getitem__, reverse=True)
            if len(rank)>1:
                if rank[0]==line[1].strip():
                    rank = rank[1:]
            ranks.append(rank)
        return ranks

    def save(self, userId):
        logger.info("Saving new model for user {}".format(userId))
        with open(RANKER_MODEL_PATH_TEMPLATE.format(userId), 'wb') as pf:
            # pickle.dump((self.fe, self.model), pf, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self, pf, pickle.HIGHEST_PROTOCOL)

    # def load(self, userId=None):
    #     if not userId:
    #         userId = self.userId
    #     with open(RANKER_MODEL_TEMPLATE.format(userId), 'rb') as pf:
    #         (self.fe, self.model) = pickle.load(pf)
    #     return self

    @staticmethod
    def staticload(userId):
        with open(RANKER_MODEL_PATH_TEMPLATE.format(userId), 'rb') as pf:
            return pickle.load(pf)
