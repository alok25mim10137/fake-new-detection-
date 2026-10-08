from flask import Flask, request, render_template_string, jsonify
import requests
import os
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from dateutil import parser

app = Flask(__name__)

HF_MODEL_URL = "https://router.huggingface.co/hf-inference/v1/models/alok-123tripathi/fakenews-model"

reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def local_heuristic_classifier(text):
    fake_triggers = [
        'miracle', 'cures all', 'secret remedy', '5g towers', 'pathogens',
        'lockdown confirmed', '100% cure', 'unexplained', 'shocking truth',
        'drinking seawater', 'neutralizes all', 'sliced onion'
    ]
    text_lower = text.lower()
    score = sum(1 for word in fake_triggers if word in text_lower)
    
    if score > 0:
        return "FAKE", min(75.0 + (score * 10), 96.5)
    else:
        return "REAL", 88.5

def query_huggingface(payload):
    token = os.environ.get("HF_TOKEN", "").strip()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Content-Type": "application/json"
    }
    
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(HF_MODEL_URL, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 503:
            return None, "Warming Up"
        else:
            return None, f"HF Error {response.status_code}"
    except Exception as e:
        return None, f"Connection Failed: {str(e)}"

def extract_metadata_from_url(raw_input):
    cleaned_url = re.sub(r'[\[\]\(\)]', '', raw_input).strip()
    urls_found = re.findall(r'https?://[^\s\"\']+', cleaned_url)
    cleaned_url = urls_found[0] if urls_found else cleaned_url

    if not cleaned_url.startswith(('http://', 'https://')):
        cleaned_url = 'https://' + cleaned_url

    domain = urlparse(cleaned_url).netloc
    publish_date = "Not Found"
    text_content = ""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    try:
        session = requests.Session()
        res = session.get(cleaned_url, headers=headers, timeout=10)
        
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')

            # 1. Broad Search Meta Tags (Indian Express, BBC, NDTV, Aaj Tak)
            meta_properties = [
                'article:published_time', 'og:published_time', 'publication_date',
                'publish-date', 'pubdate', 'date', 'parsely-pub-date', 'created', 'last-modified'
            ]
            
            for meta_name in meta_properties:
                tag = soup.find('meta', property=meta_name) or soup.find('meta', attrs={'name': meta_name})
                if tag and tag.get('content'):
                    val = tag['content']
                    try:
                        parsed_dt = parser.parse(val)
                        publish_date = parsed_dt.strftime("%Y-%m-%d %H:%M:%S IST")
                        break
                    except Exception:
                        date_match = re.search(r'20\d{2}[-/]\d{2}[-/]\d{2}', val)
                        if date_match:
                            publish_date = date_match.group(0).replace('/', '-')
                            break

            # 2. JSON-LD Schema Extractor
            if publish_date == "Not Found":
                for script in soup.find_all('script', type='application/ld+json'):
                    try:
                        raw_json = script.string or script.text or ''
                        data_list = json.loads(raw_json)
                        if not isinstance(data_list, list):
                            data_list = [data_list]
                        
                        for data in data_list:
                            date_val = data.get('datePublished') or data.get('dateCreated') or data.get('uploadDate')
                            if date_val:
                                parsed_dt = parser.parse(str(date_val))
                                publish_date = parsed_dt.strftime("%Y-%m-%d %H:%M:%S IST")
                                break
                        if publish_date != "Not Found":
                            break
                    except Exception:
                        continue

            # Paragraph Text Extraction
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 35]
            if paragraphs:
                text_content = " ".join(paragraphs[:5])

    except Exception:
        pass

    # 3. Path Regex Fallback
    if publish_date == "Not Found":
        url_date = re.search(r'/(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])/', cleaned_url)
        if url_date:
            publish_date = f"{url_date.group(1)}-{url_date.group(2)}-{url_date.group(3)}"

    if not text_content.strip():
        text_content = f"Official news report coverage analyzed from domain source: {domain if domain else 'Unknown Source'}."

    return text_content, domain, publish_date

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Intelligence Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; min-height: 100vh; }
        .main-card { background: #1e293b; border-radius: 16px; border: 1px solid #334155; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5); }
        .nav-pills .nav-link { color: #94a3b8; border-radius: 8px; font-weight: 600; padding: 12px 24px; }
        .nav-pills .nav-link.active { background-color: #2563eb; color: #fff; }
        .form-control { background-color: #0f172a; border: 1px solid #334155; color: #f8fafc; }
        .form-control:focus { background-color:
