from abc import ABCMeta, abstractmethod


class SimplificationPipeline(metaclass=ABCMeta):

    @abstractmethod
    def simplify_text(self, txt, startOffset=0, endOffset=None,
                      cwi=None, ranker=None):
        """

        :param txt:
        :param startOffset:
        :param endOffset:
        :param cwi:
        :param ranker:
        :return: original, replacements, sentence, sentence_offset_start,
        sentence_offset_end
        """
        raise NotImplementedError
