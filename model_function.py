# model.py
import numpy as np
import tensorflow as tf
import soundfile as sf
from tensorflow import keras
import librosa
import os

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# def load_my_model():
#     model = tf.keras.models.load_model('my_model.keras', custom_objects={'CTCLoss': CTCLoss})
#     return model

def load_and_preprocess_audio(audio_file):
    # An integer scalar Tensor. The window length in samples.
    frame_length = 256
    # An integer scalar Tensor. The number of samples to step.
    frame_step = 160
    # An integer scalar Tensor. The size of the FFT to apply.
    # If not provided, uses the smallest power of 2 enclosing frame_length.
    fft_length = 384
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=None)
    print(f"Original audio: duration = {audio.shape[0]/sr} seconds, sample rate = {sr} Hz")
    # Save the audio file in .wav format
    sf.write('converted_audio.wav', audio, sr, subtype='PCM_16')
    # Update the audio_file variable to point to the converted file
    audio_file = 'converted_audio.wav'

    # 1. Read wav file
    audio, sr = librosa.load(audio_file , sr=None)

    # 2. Resample the audio to 22050 Hz if it's not already
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

    # 3. Convert float audio to int16
    audio = (audio * np.iinfo(np.int16).max).astype(np.int16)

    # 4. Save the audio file in 16-bit PCM WAV format
    sf.write('resampled_audio.wav', audio, 22050, subtype='PCM_16')

    # 5. Read the resampled audio file
    audio, _ = librosa.load('resampled_audio.wav', sr=None)
    # print(f"Resampled audio: duration = {audio.shape[0]/sr} seconds, sample rate = {sr} Hz")

    # 6. Change type to float
    audio = tf.cast(audio, tf.float32)

    # 7. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )

    # 8. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    # 9. Normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    print(f"Spectrogram: shape = {spectrogram.shape}, min value = {tf.reduce_min(spectrogram)}, max value = {tf.reduce_max(spectrogram)}")

    return spectrogram



def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

def make_prediction(model, input_data):
    audio_data = load_and_preprocess_audio(input_data)

    batch_predictions = model.predict(np.array([audio_data]))
    batch_predictions = decode_batch_predictions(batch_predictions)
    return batch_predictions[0]



