from flask import jsonify
from lexi import BACKEND_VERSION


def make_response(status, message, **kwargs):
    response = dict()
    response["backend_version"] = BACKEND_VERSION
    response["status"] = status
    response["message"] = message
    for k, v in kwargs.items():
        response[str(k)] = v
    return jsonify(response)