import os
import re
import requests
import torch
import easyocr
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Hugging Face Trained Model
MODEL_NAME = "alok-123tripathi/fakenews-model"

print("Loading Model and Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Initialize EasyOCR Reader (English)
ocr_reader = easyocr.Reader(['en'], gpu=False)

def predict_text(text):
    if not text or not text.strip():
        return None, 0.0
    
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=-1)
    
    # 1 = REAL, 0 = FAKE
    label = "REAL" if prediction.item() == 1 else "FAKE"
    return label, round(confidence.item() * 100, 2)

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text[:1000] # Limit length for analysis
    except Exception as e:
        return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_type = request.form.get('type', 'text')
    extracted_text = ""
    
    if input_type == 'text':
        extracted_text = request.form.get('text', '')
    
    elif input_type == 'url':
        url = request.form.get('url', '')
        extracted_text = extract_text_from_url(url)
        if not extracted_text:
            return jsonify({'error': 'Could not scrape text from the provided URL.'}), 400
            
    elif input_type == 'image':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded.'}), 400
        
        file = request.files['image']
        image_bytes = file.read()
        
        # Perform OCR
        results = ocr_reader.readtext(image_bytes)
        extracted_text = " ".join([res[1] for res in results])
        
        if not extracted_text.strip():
            return jsonify({'error': 'No readable text found in image.'}), 400

    label, confidence = predict_text(extracted_text)
    
    return jsonify({
        'status': 'success',
        'label': label,
        'confidence': f"{confidence}%",
        'extracted_text': extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
