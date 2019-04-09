from abc import ABCMeta, abstractmethod


class SimplificationPipeline(metaclass=ABCMeta):

    @abstractmethod
    def simplify_text(self, txt, startOffset=0, endOffset=None,
                      cwi=None, ranker=None):
        raise NotImplementedError
