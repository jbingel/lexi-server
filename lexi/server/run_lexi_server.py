import argparse
import copy
import getpass
import logging
import os
import traceback
from configparser import ConfigParser
from logging.handlers import TimedRotatingFileHandler
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sslify import SSLify
from lexi.server.util.database import DatabaseConnection, \
    DatabaseConnectionError
from werkzeug.exceptions import HTTPException

from lexi.config import LEXI_BASE, LOG_DIR, RANKER_MODEL_PATH_TEMPLATE, \
    CWI_MODEL_PATH_TEMPLATE, MODELS_DIR, RESOURCES
from lexi.core.endpoints import update_ranker
from lexi.core.simplification.lexical import LexicalSimplificationPipeline, \
    LexiCWI, LexiRanker, LexiGenerator
from lexi.server.util import statuscodes
from lexi.server.util.html import process_html
from lexi.server.util.communication import make_response
# from lexi.lib.lib import OnlineRegressionRanker

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

# ARG PARSING
description = "Run a Lexi server."
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-S', '--disable-ssl',
                    action="store_true",
                    default=False,
                    help="Whether to disable SSL connection (e.g. "
                         "when developing on localhost).")
parser.add_argument('-l', '--log-level',
                    dest="log_level",
                    choices=('error', 'info', 'debug'),
                    default='info',
                    help='Logging verbosity level')
args = parser.parse_args()

os.makedirs(MODELS_DIR, exist_ok=True)

# LOGGING
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('lexi')
log_level = logging.INFO
# get logging level from CL argument, if set
if hasattr(args, 'log_level'):
    arg2level = {'error': logging.ERROR,
                 'debug': logging.DEBUG,
                 'info': logging.INFO}
    log_level = arg2level.get(args.log_level, logging.INFO)
logger.setLevel(log_level)
fh = TimedRotatingFileHandler(LOG_DIR+'/lexi.log', when="midnight",
                              interval=1, encoding="UTF-8")
fh.suffix = "%Y-%m-%d"
fh.setLevel(log_level)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(log_level)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - '
                              '{%(filename)s:%(lineno)d} '
                              '%(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# CONFIGS
cfg = ConfigParser()
fs = cfg.read(LEXI_BASE+"/lexi.cfg")


# DB CONNECTION
db_params = {
    "dbname": cfg["postgres"]["dbname"],
    "user": cfg["postgres"]["user"],
    "host": cfg["postgres"].get("host"),
    "port": int(cfg["postgres"]["port"])
}

pwd = cfg["postgres"].get("password")
if pwd:
    logger.debug(pwd)
    db_params["password"] = pwd
else:
    logger.info("No password given")

db_connected = False
db_connection = None
while not db_connected:
    try:
        db_connection = DatabaseConnection(db_params)
        db_connected = True
    except DatabaseConnectionError:
        logger.error("Couldn't connect to database. "
                     "Password wrong or not set?")
        pw = getpass.getpass("Type database password: ")
        db_params["password"] = pw


# FLASK SETUP
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

if not hasattr(args, "disable_ssl") or args.disable_ssl is False:
    logger.info("Using SSL connection.")
    sslify = SSLify(app)
else:
    logger.info("Not using SSL connection.")
app.debug = False


# LOADING DEFAULT MODEL
simplification_pipeline = LexicalSimplificationPipeline("default")
generator = LexiGenerator(synonyms_files=RESOURCES["da"]["synonyms"],
                          embedding_files=RESOURCES["da"]["embeddings"])
simplification_pipeline.setGenerator(generator)
# default_ranker = load_pickled_model(
#     RANKER_MODEL_PATH_TEMPLATE.format("default"))
# default_cwi = load_pickled_model(
#     CWI_MODEL_PATH_TEMPLATE.format("default"))
default_ranker = LexiRanker("default")
default_cwi = LexiCWI("default")  # TODO pretrain offline and load
personalized_rankers = {"default": default_ranker}
personalized_cwi = {"default": default_cwi}
logger.debug("Default ranker:")
logger.debug(type(default_ranker))
logger.debug(default_ranker)
logger.info("Base simplifier loaded.")

# BLACKLISTED WORDS, not to be simplified
GENERIC_BLACKLIST = db_connection.get_blacklist(None)
logger.debug("Generic blacklist: {}".format(GENERIC_BLACKLIST))


@app.errorhandler(Exception)
def handle_error(e):
    code = statuscodes.UNKNOWN_ERROR
    if isinstance(e, HTTPException):
        code = e.code
    logger.error(e)
    logger.error(traceback.format_exc())
    return jsonify(error=str(e)), code


@app.route("/simplify", methods=["POST"])
def process():
    if not db_connection.test_connection():
        msg = "Could not connect to database."
        logger.error(msg)
        return make_response(statuscodes.DATABASE_CONNECTION_ERROR, msg)
    email = request.json["email"].lower()
    website_url = request.json["url"]
    language = request.json.get("languageId")
    frontend_version = request.json.get("frontend_version", "N/A")
    logger.info("Received simplification request from user {} for website {}"
                .format(email, website_url))
    logger.debug("Simplification request: {}".format(request.json))
    user_id = db_connection.get_user(email)
    if not user_id:
        user_id = 1  # default user. TODO issue warning here to user
    request_id = db_connection.insert_session(user_id, website_url,
                                              frontend_version=frontend_version,
                                              language=language)

    cwi = None
    single_word_request = request.json.get("single_word_request", False)
    if not single_word_request:
        cwi = get_personalized_cwi(user_id)

    ranker = get_personalized_ranker(user_id)

    logger.info("Loaded CWI: "+str(cwi))
    logger.info("Loaded ranker: "+str(ranker))
    min_similarity = request.json.get("min_similarity", 0.65)
    if not type(min_similarity) == float:
        raise ValueError("'min_similarity' must be a float. You "
                         "provided a {}".format(type(min_similarity)))
    html_out, simplifications = process_html(simplification_pipeline,
                                             request.json["html"],
                                             request.json.get("startOffset"),
                                             request.json.get("endOffset"),
                                             cwi, ranker, mode="lexical",
                                             requestId=request_id,
                                             min_similarity=min_similarity,
                                             blacklist=GENERIC_BLACKLIST)
    db_connection.update_session_with_simplifications(request_id,
                                                      simplifications)
    return make_response(statuscodes.OK,
                         "Simplification successful",
                         html=html_out,
                         simplifications=simplifications,
                         request_id=request_id)


@app.route("/register_user", methods=["POST"])
def register_user():
    if not db_connection.test_connection():
        msg = "Could not connect to database."
        logger.error(msg)
        return make_response(statuscodes.DATABASE_CONNECTION_ERROR, msg)
    # get fields from request
    email = request.json["email"].lower()
    # pw_hash = request.json["pw_hash"]
    year_of_birth = request.json["year_of_birth"]
    education = request.json["education"]
    # get maximum user ID
    logger.info("New user: {}".format([email, year_of_birth, education]))
    user = db_connection.get_user(email)
    if user:
        msg = "Email address {} already registered".format(email)
        logger.info(msg)
        return make_response(statuscodes.EMAIL_ADDRESS_REGISTERED, msg)
    else:
        new_user_id = db_connection.insert_user(email)
        model_path = RANKER_MODEL_PATH_TEMPLATE.format(new_user_id)
        db_connection.insert_model(new_user_id, year_of_birth, education,
                                   model_path, "ranker")
        return make_response(statuscodes.OK, "Registration successful")


@app.route("/login", methods=["POST"])
def login_user():
    if not db_connection.test_connection():
        msg = "Could not connect to database."
        logger.error(msg)
        return make_response(statuscodes.DATABASE_CONNECTION_ERROR, msg)
    email = request.json["email"].lower()
    logger.info("Received login request for user {}".format(email))
    user_id = db_connection.get_user(email)
    if not user_id:
        msg = "Email address {} not found".format(email)
        logger.info(msg)
        return make_response(statuscodes.EMAIL_ADDRESS_NOT_FOUND, msg)
    msg = "OK. Logged on with email {}".format(email)
    logger.info(msg)
    # db_connection.update_last_login(user_id)
    return make_response(statuscodes.OK, "Login successful")


@app.route("/feedback", methods=["POST"])
def get_feedback():
    email = request.json["email"].lower()
    user_id = db_connection.get_user(email)
    simplifications = request.json.get("simplifications", None)
    feedback_text = request.json.get("feedback_text", "N/A")
    website = request.json.get("url", "N/A")
    # request_id = request.json.get("request_id", -1)
    rating = request.json.get("rating", 0)
    logger.info("Received feedback from user {}, rating {} for site {}.\n"
                "Feedback text: {}".format(email, rating, website,
                                           feedback_text))
    logger.debug("Request: {}".format(request.json))
    db_connection.update_session_with_feedback(rating, feedback_text,
                                               simplifications)
    if simplifications:
        logger.info(simplifications)
        # try:
        logger.debug("Getting ranker for user: {}".format(user_id))
        ranker = personalized_rankers[user_id]
        logger.debug("Ranker: {}".format(ranker))
        update_ranker(ranker, user_id, simplifications, rating)
        return make_response(statuscodes.OK, "Feedback successful")
    else:
        msg = "Did not receive any simplificiations"
        logger.info(msg)
        return make_response(statuscodes.NO_SIMPLIFICATIONS_RETURNED, msg)


@app.route("/test_connection", methods=["POST"])
def test_connection():
    logger.debug("Connection test from user {}".format(request.json["email"]))
    working_fine = db_connection.test_connection()
    if working_fine:
        return make_response(statuscodes.OK, "Working fine.")
    else:
        return make_response(statuscodes.DATABASE_CONNECTION_ERROR,
                             "Something is wrong with the database.")


@app.route("/versioncheck", methods=["POST"])
def versioncheck():
    frontend_most_recent_version = cfg["frontend"]["most_recent_version"]
    download_url = cfg["frontend"]["download_url"]
    return make_response(statuscodes.OK,
                         "Returned most recent frontend version",
                         most_recent_version=frontend_most_recent_version,
                         download_url=download_url)


def get_personalized_ranker(user_id):
    if user_id in personalized_rankers:
        logger.info("Using personalized ranker, still in memory.")
        ranker = personalized_rankers[user_id]
    else:
        logger.info("Gotta load ranker or use default...")
        try:
            # retrieve model
            model_path = db_connection.get_model_path(user_id)
            ranker = LexiRanker(user_id)
        except:
            logger.warning("Could not load personalized model. "
                           "Loading default ranker.")
            ranker = copy.copy(personalized_rankers["default"])
            logger.debug(ranker)
            ranker.userId = user_id
        personalized_rankers[user_id] = ranker
    return ranker


def get_personalized_cwi(user_id):
    if user_id in personalized_cwi:
        logger.info("Using personalized cwi, still in memory.")
        cwi = personalized_cwi[user_id]
    else:
        logger.info("Gotta load cwi or use default...")
        try:
            # retrieve model
            model_path = db_connection.get_model_path(user_id)
            cwi = LexiCWI(user_id)
        except:
            logger.warning("Could not load personalized model. "
                           "Loading default cwi.")
            cwi = copy.copy(personalized_cwi["default"])
            logger.debug(cwi)
            cwi.userId = user_id
        personalized_cwi[user_id] = cwi
    return cwi


if __name__ == "__main__":
    app.run(threaded=True)
    logger.debug("Rules: " + str([rule for rule in app.url_map.iter_rules()]))
