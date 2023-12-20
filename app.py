from flask import Flask, request, jsonify
from model_function import CTCLoss , make_prediction
import numpy as np
import tensorflow as tf
from google.cloud import storage
import os
from io import BytesIO

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
storage_client = storage.Client()
model = tf.keras.models.load_model('my_model.keras', custom_objects={'CTCLoss': CTCLoss})
# layer = model.get_layer('conv_1')

@app.route('/api', methods=[ 'GET' , 'POST'])
def predict():
    # Get data from Post request
    if request.method == 'POST':
        try:
            audio_bucket = storage_client.get_bucket(
                'anna_app_bucket')
            filename = request.json['filename']
            audio_blob = audio_bucket.blob('audio/' + filename)
            audio_path = BytesIO(audio_blob.download_as_bytes())
        except Exception:
            respond = jsonify({'message': 'Error loading audio file'})
            respond.status_code = 400
            return respond

        prediction = make_prediction(model , audio_path )

        return jsonify(prediction)
    return 'OK'


if __name__ == '__main__':
    app.run(port=5000, debug=True)
