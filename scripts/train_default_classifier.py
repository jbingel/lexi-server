from lexi.config import RESOURCES
from lexi.core.simplification.lexical import *


def fresh_train(userId="default", language="da", resources=None):
    c = LexicalSimplificationPipeline(userId=userId, language=language)
    if not resources:
        try:
            resources = RESOURCES[language]
        except KeyError:
            print("Couldn't find resources for language {}".format(language))

    # Generator
    g = LexiGenerator(synonyms_files=resources["synonyms"],
                      embedding_files=resources["embeddings"])
    c.setGenerator(g)

    # Ranker
    c.setRanker(LexiRanker("default"))
    return c


c = fresh_train()

c.ranker.save("default")
c.cwi.save("default")
