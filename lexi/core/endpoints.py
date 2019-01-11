import logging

from lexi.core.util import util

logger = logging.getLogger('lexi')
parser = None


def process_html_structured(classifier, html, ranker, parId):
    """
    Transforms HMTL source, enriching simplified text spans with core markup by
    separating markup from text and sending pure text to simplification class.

    :param classifier: Simplification classifier instance
    :param html: Input HTML source
    :param parId: Paragraph identifier to disambiguate core simplification
    targets across multiple calls to this method
    :return: a tuple containing (1) the enriched HTML output and (2) a set of
    dicts that map core simplification target IDs to dicts of the form:
    ```
    {
        "original": original,
        "simple": simple,
        "bad_feedback": False,
        "is_simplified": False}
    }
    ```
    """
    global parser
    if not parser:
        from pylinkgrammar.linkgrammar import Parser
        parser = Parser()
    simplifications = {}
    html_out = []
    spanId = 0
    if not html.strip():
        return html
    output_sents = classifier.predict_text(html, ranker)
    for original, simple in zip(*output_sents):
        simple_parsed = parser.parse_sent(simple)
        logger.debug([simple_parsed, simple.replace('\n', ''), parser])
        logger.debug(original)
        if original == simple:
            html_out.append(original)
        # elif not simple_parsed:
        #     out.append(original)
        else:
            original = util.escape(original)
            simple = util.escape(simple)
            spanId += 1
            elemId = "lexi_{}_{}".format(parId, spanId)
            html_out.append("<span id='{}' class='lexi-simplify'>{}</span>".
                            format(elemId, original))
            simplifications.update(
                {elemId: {
                    "original": original,
                    "simple": simple,
                    "bad_feedback": False,
                    "is_simplified": False,
                    # "sentence": sentence,  # uncomment if ever needing this
                    # "index": word_index
                }
                })
    return " ".join(html_out), simplifications


def process_html_lexical(classifier, html, startOffset, endOffset, ranker,
                         requestId=0, min_similarity=0.7,
                         blacklist=None):
    """
    Transforms HMTL source, enriching simplified words with core markup by
    separating markup from text and sending pure text to simplification class.

    :param classifier: Simplification classifier instance
    :param html: Input HTML source
    :param startOffset: offset after which simplifications are solicited
    :param endOffset: offset until which simplifications are solicited
    :param ranker: personalized ranker
    :param requestId: Request identifier to disambiguate core simplification
    targets across multiple calls to this method
    :param min_similarity: minimum similarity score for candidates, if
    applicable
    :param blacklist: list of words not to be simplified
    :return: a tuple containing (1) the enriched HTML output and (2) a set of
    dicts that map core simplification target IDs to dicts of the form:
    ```
    {
        "original": original word,
        "simple": simplified word,
        "choices": list of word alternatives available for display
        "bad_feedback": boolean, will be filled by frontend indicating bad
                        feedback,
        "selection": integer, will be filled by frontend, storing the number of
                     clicks by user to arrive at ultimately selected alternative
        "sentence": the original sentence string (without markup),
        "word_index": integer, index of the target word in the whitespace-
                      separated sentence string (counting from 0)
    }
    ```
    """
    def get_local_hyperlink_balance(tags):
        local_hyperlink_balance = 0
        for tag in tags:
            if tag.startswith("<a "):
                local_hyperlink_balance += 1
            elif tag == "</a>":
                local_hyperlink_balance -= 1
        return local_hyperlink_balance

    simplifications = {}
    html_out = ""
    spanId = 0
    if not html.strip():
        return html, simplifications
    # output is a sequence of tokens including whitespaces, id2simplification
    # is a dict mapping token IDs to simplifications, if applicable
    offset2html, pure_text = util.filter_html(html)
    offset2simplification = classifier.predict_text(
        pure_text, startOffset, endOffset, ranker,
        min_similarity=min_similarity, blacklist=blacklist)
    logger.debug("Simplifying text between character offsets {} "
                 "and {}: {}".format(startOffset, endOffset, pure_text))
    i = 0
    open_hyperlinks_count = 0
    while i < len(pure_text):
        tags_at_offset = offset2html.get(i, [])
        open_hyperlinks_count += get_local_hyperlink_balance(tags_at_offset)
        # insert any HTML markup that belongs here
        if i in offset2html:
            html_out += "".join(offset2html[i])
        if i in offset2simplification and not open_hyperlinks_count > 0:
            # checking for hyperlinks because we don't want to simplify those
            original, simple, sentence, word_index = offset2simplification[i]
            # in future, possibly get more alternatives, and possibly return
            # in some other order
            choices = [original, simple]
            simple = util.escape(simple)
            spanId += 1
            elemId = "lexi_{}_{}".format(requestId, spanId)
            displaying_original = "true" if choices[0] == original else "false"
            span_out = "<span id='{}' " \
                       "class='lexi-simplify' " \
                       "data-displaying-original='{}'>" \
                       "{}" \
                       "</span>"\
                .format(elemId, displaying_original, original)
            html_out += span_out
            # TODO allow more than two alternatives as `choices' in future
            # (https://github.com/jbingel/lexi-backend/issues/2)
            simplifications.update(
                {elemId: {
                    "request_id": requestId,
                    "original": original, 
                    "simple": simple,  # legacy for frontend version <= 0.2
                    "choices": choices,
                    "bad_feedback": False,
                    "selection": 0,
                    "sentence": sentence,
                    "word_index": word_index,
                }
                })
            i += len(original)-1
        else:
            html_out += pure_text[i]
        i += 1
    if i in offset2html:  # Append possible final markup at offset len(text)+1
        html_out += "".join(offset2html[i])
    return html_out, simplifications


def update_classifier(classifier, feedback):
    """
    Featurizes simplification feedback from user and updates classifier
    accordingly
    :param classifier:
    :param feedback:
    :return:
    """
    xs, ys = [], []
    for item in feedback.values():
        original = item["original"]
        simple = item["simple"]
        original_is_simpler = item["is_simplified"]  # boolean
        xs.append(original)
        ys.append(int(original_is_simpler))  # 1 iff original_is_simpler
        xs.append(simple)
        ys.append(int(not original_is_simpler))  # inverse
    try:
        classifier.featurize_update(xs, ys)
    except AttributeError:  # likely because featurizer hasn't been trained
        classifier.featurize_train(xs, ys)


def update_ranker(ranker, user_id, feedback, overall_rating=0):
    """
    Collects feedback and updates ranker
    :param ranker: The personal ranker to update
    :param user_id: ID for this user
    :param feedback: The simplification choices and feedback from the user
    :param overall_rating: a 1-to-5 scale rating of the overall performance
    :return:
    """
    # ranker online training expects formatted text as input (instead of
    # structured data)
    # TODO really gotta improve interface, make ranker take structured data
    textblock = ""
    for _, simplification in feedback.items():
        if simplification["bad_feedback"]:
            continue
        selection = simplification["selection"]
        choices = simplification.get("choices")
        if not choices:
            logger.warning("No `choices` field in the "
                           "simplifications: {}".format(simplification))
            return None
        logger.debug(simplification)
        if selection == 0:  # nothing selected
            continue
        else:
            simple_index = selection % len(choices)
            simple_word = choices[simple_index]
            difficult_words = [w for w in choices if not w == simple_word]
            for difficult in difficult_words:
                textblock += "{}\t{}\t{}\t{}\t{}\n".format(
                    simplification["sentence"].replace("\n", " "),
                    choices,
                    str(simplification["word_index"]),
                    "1:" + simple_word,
                    "2:" + difficult
                )
    textblock = textblock.strip()  # remove last newline
    if textblock:
        logger.debug("Updating with the following textblock:\n\n"+textblock)
        ranker.onlineTrainRegressionModel(textblock)
        ranker.save(user_id)
    else:
        logger.info("Didn't get any useable feedback.")
