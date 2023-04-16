import os
from logging.config import dictConfig

from flask import Flask, request, jsonify
from flask_cors import CORS
from pyannote.audio import Pipeline
from werkzeug.exceptions import InternalServerError
from dotenv import load_dotenv

import classifier as clf
import config
import model
import preprocess as pre

load_dotenv()

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

try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=os.getenv("USE_AUTH_TOKEN_HF"))
except Exception as e:
    raise InternalServerError(f"Failed to connect with Pipeline to download pre trained model")


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
        # Dictionary to store speaker information
        speaker_dict = {}

        # Create the directory for saving the models weights if not exist
        os.makedirs('audio_files', exist_ok=True)

        audio_file.save('audio_files/{}'.format(audio_file.filename))
        diarization = pipeline('audio_files/{}'.format(audio_file.filename))
        audio, sample_rate = pre.load_audio('audio_files/{}'.format(audio_file.filename))

        # iterate over speaker turns and plot spectrogram's for each turn and compute the emotion
        for turn, _, speaker in diarization.itertracks(yield_label=True):

            print("========================================================")
            print()

            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s {speaker}")

            # Get start and end time of the turn
            start, end = turn.start, turn.end

            # Crop the audio signal for the turn
            audio_split = pre.split_audio(audio, sample_rate, start, end)

            print()

            spect_img = pre.get_audio_spectrogram_image(audio_split,sample_rate)
            prediction_results = model.get_prediction(img=spect_img, model=MODEL)
            classification_report = clf.get_detection_report(prediction_results)

            print(classification_report)

            # Add the speaker, start, end times, and spectrogram image to the dictionary
            if speaker not in speaker_dict:
                speaker_dict[speaker] = []

            speaker_dict[speaker].append({
                "start_time": str(start)+' s',
                "end_time": str(end)+' s',
                "classification_report": classification_report
            })

    except IndexError as e:
        return {"error": str(e)}, 400

    except Exception as e:
        return {"error": str(e)}, 500

    return jsonify(speaker_dict), 200


if __name__ == "__main__":
    app.run(debug=True)
