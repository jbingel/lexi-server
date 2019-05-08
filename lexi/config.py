import os

SOURCE_BASE = os.path.dirname(os.path.realpath(__file__))
LEXI_BASE = os.path.join(SOURCE_BASE, "..")
LOG_DIR = os.path.join(LEXI_BASE, "logs")
MODELS_DIR = os.path.join(LEXI_BASE, "models")
RANKER_DIR = os.path.join(MODELS_DIR, "rankers")
CWI_DIR = os.path.join(MODELS_DIR, "cwi")
SCORERS_DIR = os.path.join(MODELS_DIR, "scorers")
RESOURCES_DIR = os.path.join(LEXI_BASE, "res")
STANFORDNLP = os.path.join(RESOURCES_DIR, "stanfordnlp_resources")

RANKER_PATH_TEMPLATE = os.path.join(RANKER_DIR, "{}.json")
CWI_PATH_TEMPLATE = os.path.join(CWI_DIR, "{}.json")
SCORER_PATH_TEMPLATE = os.path.join(SCORERS_DIR, "{}.json")
SCORER_MODEL_PATH_TEMPLATE = os.path.join(SCORERS_DIR, "{}.pt")

LEXICAL_MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "{}-lexical.pickle")
MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "{}.pickle")

RESOURCES_FULL = {
    "da": {
        "embeddings":
            [RESOURCES_DIR+"/da/embeddings/danish_word_vectors_1300_cbow.bin",
             RESOURCES_DIR+"/da/embeddings/da.bin"],
        "lm":
            RESOURCES_DIR+"/da/lm/danish_lm.bin",
        "ubr":
            RESOURCES_DIR+"/da/simplification/danish_dataset_ubr.txt",
        "ranking_training_dataset":
            RESOURCES_DIR+"/da/simplification/clean_danish_ls_dataset.txt",
        "synonyms":
            [RESOURCES_DIR + "/da/synonyms/da_synonyms_combined.csv"]}
}

RESOURCES = {
    "da": {
        "embeddings":
            [RESOURCES_DIR+"/da/embeddings/danish_word_vectors_1300_cbow_"
                           "filtered.bin",
             RESOURCES_DIR + "/da/embeddings/da.bin"],
        "lm":
            RESOURCES_DIR + "/da/lm/danish_lm.bin",
        "ubr":
            RESOURCES_DIR + "/da/simplification/danish_dataset_ubr.txt",
        "ranking_training_dataset":
            RESOURCES_DIR + "/da/simplification/clean_danish_ls_dataset.txt",
        "synonyms":
            [RESOURCES_DIR + "/da/synonyms/da_synonyms_combined.csv"]}
}
