import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def get_audio_spectrogram_image(audio_clip, sample_rate):
    y = audio_clip
    sr = sample_rate

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


def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate


def split_audio(audio, sample_rate, start_time, end_time):
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)
    return audio[start_frame:end_frame]
