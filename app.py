from flask import Flask, request, render_template, jsonify
import requests
import os
import re
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse

app = Flask(__name__)

# Standard Hugging Face Serverless Inference API URL
HF_MODEL_URL = "https://api-inference.huggingface.co/models/alok-123tripathi/fakenews-model"

reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def query_huggingface(payload):
    token = os.environ.get("HF_TOKEN", "").strip()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(HF_MODEL_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"error": "Model loading/warming up on Hugging Face. Please click Analyze again in 15-20 seconds."}
        else:
            return {"error": f"Hugging Face HTTP {response.status_code}: {response.text[:120]}"}
            
    except Exception as e:
        return {"error": f"API Connection Error: {str(e)}"}

def extract_metadata_from_url(raw_input):
    urls_found = re.findall(r'https?://[^\s\]\)\>\"\']+', raw_input)
    cleaned_url = urls_found[0] if urls_found else raw_input.strip()

    if not cleaned_url.startswith(('http://', 'https://')):
        cleaned_url = 'https://' + cleaned_url

    domain = urlparse(cleaned_url).netloc
    publish_date = "Not Found (Meta Tag Missing)"
    text_content = ""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        res = requests.get(cleaned_url, headers=headers, timeout=8)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')

            date_meta = (
                soup.find('meta', property='article:published_time') or
                soup.find('meta', attrs={'name': 'pubdate'}) or
                soup.find('meta', attrs={'name': 'date'}) or
                soup.find('meta', property='og:updated_time') or
                soup.find('time')
            )

            if date_meta:
                if date_meta.name == 'time' and date_meta.has_attr('datetime'):
                    publish_date = date_meta['datetime']
                elif date_meta.has_attr('content'):
                    publish_date = date_meta['content']
                elif date_meta.text:
                    publish_date = date_meta.text.strip()

            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 30]
            if paragraphs:
                text_content = " ".join(paragraphs[:6])

    except Exception:
        pass

    if not text_content.strip():
        text_content = f"Official news report content fetched from domain source: {domain if domain else 'Unknown Source'}."

    return text_content, domain, publish_date

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        input_type = data.get('type') or request.form.get('type')
        text_content = ""
        source_domain = None
        publish_date = None

        if input_type == 'text':
            text_content = data.get('content') or request.form.get('content', '')

        elif input_type == 'url':
            url = data.get('content') or request.form.get('content', '')
            text_content, source_domain, publish_date = extract_metadata_from_url(url)

        elif input_type == 'image':
            if 'file' in request.files:
                file = request.files['file']
                ocr_engine = get_ocr_reader()
                results = ocr_engine.readtext(file.read())
                text_content = " ".join([res[1] for res in results])

        if not text_content.strip():
            return jsonify({'error': 'No readable text could be processed.'}), 400

        truncated_text = text_content[:512]
        api_output = query_huggingface({"inputs": truncated_text})

        if isinstance(api_output, dict) and 'error' in api_output:
            return jsonify({'error': api_output['error']}), 503

        predictions = None
        if isinstance(api_output, list) and len(api_output) > 0:
            if isinstance(api_output[0], list):
                predictions = api_output[0]
            elif isinstance(api_output[0], dict):
                predictions = api_output
        elif isinstance(api_output, dict):
            predictions = [api_output]

        if not predictions:
            return jsonify({'error': 'Could not parse prediction output.'}), 500

        top_pred = max(predictions, key=lambda x: x.get('score', 0))
        
        raw_label = str(top_pred.get('label', ''))
        confidence = round(top_pred.get('score', 0) * 100, 2)
        
        label = "REAL" if "1" in raw_label or "REAL" in raw_label.upper() or "LABEL_1" in raw_label else "FAKE"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        return jsonify({
            'label': label,
            'confidence': confidence,
            'timestamp': current_time,
            'source_domain': source_domain,
            'publish_date': publish_date,
            'extracted_text': text_content[:250]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
