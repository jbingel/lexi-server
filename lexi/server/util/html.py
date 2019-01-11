from lexi.core.endpoints import process_html_lexical


def map_text_to_html_offsets(html_src):
    """
    Maps text offsets to HTML offsets, e.g. in HTML source `<p>One moring, when
    <a href="index.html">Georg Samsa</a> woke</p>`, text offset 0 (before 'One')
    is mapped to HTML offset 3 (after `<p>`).
    :param html_src: The HTML source in question
    :return: a dictinary mapping all text offsets to their corresponding HTML
    offset
    """
    mapping = {}
    text_idx = 0
    inside_tag = False
    for i, c in enumerate(html_src):
        if c == "<":
            inside_tag = True
        elif c == ">":
            inside_tag = False
        elif not inside_tag:
            mapping[text_idx] = i
            text_idx += 1
    return mapping


def process_html(classifier, html_src, startOffset, endOffset, ranker,
                 mode="lexical", requestId=0, min_similarity=0.7,
                 blacklist=None):
    """
    :param classifier:
    :param html_src: The HTML source in question
    :param ranker: Ranker to use with this classifier
    :param mode: simplification mode (whether to perform lexical simplification,
     sentence simplification, ...). Only "lexical" accepted for now.
    :param requestId: Request identifier to disambiguate core simplification
    targets across multiple calls to this method
    :param min_similarity: minimum similarity for replacements, if applicable
    :param blacklist: list of words not to be simplified
    :return: processed text
    """
    simplifications = {}
    html_out = ""
    if mode == "lexical":
        _output, _simplifications = process_html_lexical(
            classifier, html_src, startOffset, endOffset, requestId=requestId,
            ranker=ranker,
            min_similarity=min_similarity,
            blacklist=blacklist)
    else:
        # _output, _simplifications = process_html_structured(
        #     classifier, html_src, ranker, 0)
        raise NotImplementedError("Only 'lexical' simplification mode "
                                  "implemented so far. You specified {}.".
                                  format(mode))
    html_out += _output
    simplifications.update(_simplifications)
    return html_out, simplifications

#
# def process_html(classifier, html_src, lexical=True):
#     """
#     :param classifier:
#     :param html_src: The HTML source in question
#     :param lexical: whether to perform lexical simplification only
#     :return: processed text
#     """
#     simplifications = {}
#     html_out = ""
#     inside_tag = False
#     allowed_tags = ["p", "a", "strong", "emph", "i", "b"]
#     textbuffer = ""
#     parId = 0
#     cur_tag = ""
#     for i, c in enumerate(html_src):
#         if c == "<":
#             inside_tag = True
#             parId += 1
#             if cur_tag in allowed_tags:
#                 print(textbuffer)
#                 if lexical:
#                     _output, _simplifications = process_text_lexical(
#                         classifier, textbuffer, parId)
#                 else:
#                     _output, _simplifications = process_text_sentences(
#                         classifier, textbuffer, parId)
#                 print(_simplifications)
#                 html_out += _output
#                 simplifications.update(_simplifications)
#             else:
#                 html_out += textbuffer
#             textbuffer = ""
#             cur_tag = ""
#             html_out += c
#         elif c == ">":
#             inside_tag = False
#             html_out += c
#         elif inside_tag:
#             cur_tag += c
#             html_out += c
#         else:  # text content
#             textbuffer += c
#     return html_out, simplifications
