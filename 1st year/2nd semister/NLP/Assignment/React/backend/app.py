from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from stt_model import STTModel
from loader import DataLoader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Initialize the loader and model
    # Current working directory
cwd = os.getcwd()
script_dir = os.path.dirname(__file__)
print('directory: %s' % script_dir)
data_dit = '\data\\train\wav'
loader = DataLoader(script_dir + data_dit)
model = STTModel(loader)

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    epochs = data.get('epochs', 20)
    batch_size = data.get('batch_size', 20)
    
    input_to_softmax = model.model(input_dim=13, filters=200, kernel_size=11, conv_stride=2, conv_border_mode='valid',
                                   units=200, output_dim=29)
    
    model.train(model.next_train(batch_size), input_to_softmax, model_name='your_model.h5', minibatch_size=batch_size, epochs=epochs)
    return jsonify({'message': 'Model trained successfully'})

@app.route('/test', methods=['POST'])
def test():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Use the model to predict the transcription
        feature = model.featurize(filepath)
        normalized_feature = model.normalize(feature)
        input_data = tf.expand_dims(normalized_feature, axis=0)
        
        # Load the trained model
        input_to_softmax = model.model(input_dim=13, filters=200, kernel_size=11, conv_stride=2, conv_border_mode='valid',
                                       units=200, output_dim=29)
        input_to_softmax.load_weights('models/your_model.h5')
        
        # Make predictions
        predictions = input_to_softmax.predict(input_data)
        # Decode the predictions (this part depends on your implementation of decoding)
        transcription = decode_predictions(predictions)
        
        return jsonify({'transcription': transcription})

def decode_predictions(predictions):
    # Implement your decoding logic here
    # This is a placeholder function
    return 'decoded transcription'

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
