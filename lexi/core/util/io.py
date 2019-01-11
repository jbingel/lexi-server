import pickle
import logging
logger = logging.getLogger('core')


def load_pickled_model(filename):
    logger.info("Loading pickled file {}".format(filename))
    with open(filename, 'rb') as pickled_data:
        model = pickle.load(pickled_data)
    logger.info("Done loading pickled file {}".format(filename))
    return model


def load_pickled(filename):
    with open(filename, 'rb') as pickled_data:
        X, y = pickle.load(pickled_data)
    logger.debug("Loaded data with %d instances" % (len(X)))
    return X, y