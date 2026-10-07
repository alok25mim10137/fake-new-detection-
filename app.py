from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import os

# Limit PyTorch memory & threads for 512MB RAM server
torch.set_num_threads(1)

app = Flask(__name__)

# Load model and tokenizer
MODEL_NAME = "alok-123tripathi/fakenews-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Global OCR variable (Lazy Loaded)
reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

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

        # Model Inference with torch.no_grad() to save RAM
        inputs = tokenizer(text_content, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = round(probs[0][pred_class].item() * 100, 2)

        label = "REAL" if pred_class == 1 else "FAKE"
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
