from flask import Flask, request, render_template, jsonify
import requests
import os
from bs4 import BeautifulSoup

app = Flask(__name__)

# Hugging Face Inference API Endpoint
HF_MODEL_URL = "https://api-inference.huggingface.co/models/alok-123tripathi/fakenews-model"

# Lazy loaded EasyOCR
reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def query_huggingface(payload):
    # Free public inference call
    response = requests.post(HF_MODEL_URL, json=payload, timeout=20)
    return response.json()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        input_type = data.get('type') or request.form.get('type')
        text_content = ""

        if input_type == 'text':
            text_content = data.get('content') or request.form.get('content', '')

        elif input_type == 'url':
            url = data.get('content') or request.form.get('content', '')
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            text_content = " ".join(paragraphs[:5])

        elif input_type == 'image':
            if 'file' in request.files:
                file = request.files['file']
                ocr_engine = get_ocr_reader()
                results = ocr_engine.readtext(file.read())
                text_content = " ".join([res[1] for res in results])

        if not text_content.strip():
            return jsonify({'error': 'No readable text provided'}), 400

        # Truncate input text
        truncated_text = text_content[:512]

        # Query HuggingFace Serverless API
        api_output = query_huggingface({"inputs": truncated_text})

        # Handle API response structure
        if isinstance(api_output, dict) and 'error' in api_output:
            # If model is loading on HF end, wait or notify
            return jsonify({'error': 'Model is initializing on HuggingFace. Please try again in 20 seconds.'}), 503

        # Parse HF response
        predictions = api_output[0] if isinstance(api_output, list) and len(api_output) > 0 and isinstance(api_output[0], list) else api_output
        top_pred = max(predictions, key=lambda x: x['score'])
        
        # Mapping label output
        raw_label = top_pred.get('label', '')
        confidence = round(top_pred.get('score', 0) * 100, 2)
        
        label = "REAL" if "1" in raw_label or "REAL" in raw_label.upper() or "LABEL_1" in raw_label else "FAKE"

        return jsonify({
            'label': label,
            'confidence': confidence,
            'extracted_text': text_content[:300]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
