from flask import Flask, request, jsonify
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa #for loading datasets

# Initialize the Flask app
app = Flask(__name__)

# Load the tokenizer and model from Hugging Face
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

@app.route('/asr', methods=['POST'])
def asr():
    # Check if file part exists in the request
    if 'file' not in request.files:
        return jsonify({"error": "File not provided"}), 400

    # Save the uploaded file temporarily
    audio_file = request.files['file']
    temp_filename = "temp_audio.mp3"
    audio_file.save(temp_filename)
    
    try:
        # Load the audio file with librosa ensuring 16kHz sample rate
        waveform, sample_rate = librosa.load(temp_filename, sr=16000)
        
        # Calculate the duration in seconds
        duration = librosa.get_duration(y=waveform, sr=sample_rate)
        
        # Prepare the input for the model: tokenizing the waveform
        input_values = tokenizer(waveform, return_tensors="pt", padding="longest").input_values
        
        # Perform inference without gradient computation
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Decode the predicted token IDs to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        
        # Build the JSON response
        response = {
            "transcription": transcription,
            "duration": f"{duration:.1f}"  # Format duration to one decimal place as a string
        }
        
    except Exception as e:
        # In case of any errors during processing, return an error response
        response = {"error": str(e)}
        return jsonify(response), 500
    finally:
        # Clean up: remove the temporary audio file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return jsonify(response), 200

if __name__ == '__main__':
    # Run the Flask app on port 8001
    app.run(host='0.0.0.0', port=8001)
