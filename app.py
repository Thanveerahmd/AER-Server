import os
from logging.config import dictConfig

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import InternalServerError

import classifier as clf
import config
import model
import preprocess as pre

app = Flask(__name__)

# defining cors policy
cors = CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True, "methods": ["GET", "POST"]}})

# Logger configuration
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '{name} at {asctime} ({levelname}) :: {message}',
        'style': '{'
    }},
    'handlers': {
        "file": {
            "class": "logging.FileHandler",
            "filename": "dev.log",
            "formatter": "default",
            'level': 'DEBUG'
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['file']
    }
})

if not os.path.exists(config.MODEL_WEIGHT_DIR):
    os.makedirs(config.MODEL_WEIGHT_DIR)

if not os.path.exists("{}/{}".format(config.MODEL_WEIGHT_DIR, config.MODEL_WEIGHT_FILE)):
    raise InternalServerError(f"Failed to detect the model weight file. please add the model weight file in relevant "
                              f"directory  ")

try:
    MODEL = model.get_aec_model()
    MODEL.load_weights(
        "{}/{}".format(config.MODEL_WEIGHT_DIR, config.MODEL_WEIGHT_FILE))

except Exception as e:
    raise InternalServerError(f"Failed to load the model: {str(e)}")


@app.route("/", methods=["GET"])
def hello_aer_server():
    return {"msg": "Hello from AER Server"}


@app.route('/upload_audio', methods=['POST'])
def upload_audio_file():
    if not request.files:
        return {"error": "Invalid request"}, 400

    if 'audio' not in request.files:
        return {'error': "Invalid request format: missing 'audio' field"}, 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return {'error': 'No file selected for uploading'}, 400

    try:

        spect_img = pre.get_audio_spectrogram_image(audio_file)
        prediction_results = model.get_prediction(img=spect_img, model=MODEL)
        classification_report = clf.get_detection_report(prediction_results)

    except IndexError as e:
        return {"error": str(e)}, 400

    except Exception as e:
        return {"error": str(e)}, 500

    return jsonify(classification_report), 200


if __name__ == "__main__":
    app.run(debug=True)
