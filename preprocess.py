import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_audio_spectrogram_image(audio_clip):
    # load the audio file with librosa
    y, sr = librosa.load(audio_clip.stream)

    # Generate a spectrogram
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    spect_db = librosa.power_to_db(spect, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Convert the spectrogram image to a PIL Image format
    img_file = io.BytesIO()
    plt.savefig(img_file, format='png')
    img_file.seek(0)
    img = Image.open(img_file)

    # Convert the image to RGB mode and resize
    img = img.convert('RGB')
    img = img.resize((299, 299))

    return img
