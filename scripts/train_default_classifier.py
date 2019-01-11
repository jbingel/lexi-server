from lexi.config import RESOURCES, RESOURCES_TEST
from lexi.core.simplification.lexical import *


def fresh_train(userId="default", language="da", resources=None):
    c = LexensteinSimplifier(userId=userId, language=language)
    if not resources:
        try:
            #resources = RESOURCES[language]
            resources = RESOURCES_TEST[language]
            print("WARNING: CHECK FOR CORRECT RESOURCES! (using test)")
        except KeyError:
            print("Couldn't find resources for language {}".format(language))
    # General purpose
    w2vpm = resources['embeddings']
    # Generator
    # gg = LexensteinGenerator(w2vpm)
    gg = SynonymDBGenerator(resources['synonyms'])
    # gg = LexensteinGenerator(w2vpm)

    # Selector
    fe = FeatureEstimator()
    # fe.resources[w2vpm[0]] = gg.model
    fe.addCollocationalFeature(resources['lm'], 2, 2, 'Complexity')
    fe.addWordVectorSimilarityFeature(w2vpm[0], 'Simplicity')
    br = BoundaryRanker(fe)
    bs = BoundarySelector(br)
    bs.trainSelectorWithCrossValidation(resources['ubr'], 1, 5, 0.25, k='all')
    # Ranker
    fe = FeatureEstimator()
    fe.addLengthFeature('Complexity')
    fe.addCollocationalFeature(resources['lm'], 2, 2, 'Simplicity')
    orr = OnlineRegressionRanker(fe, None, training_dataset=resources[
        'ranking_training_dataset'])
    # Return LexicalSimplifier object
    c.generator = gg
    c.selector = bs
    c.ranker = orr
    return c

c = fresh_train()
c.save()

r = c.ranker
r.save("default")
