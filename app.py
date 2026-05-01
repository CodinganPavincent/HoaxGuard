from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import pickle
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from duckduckgo_search import DDGS

app = Flask(__name__)

# Kredibilitas Media
CREDIBLE_HIGH = ['kompas', 'tempo', 'cnn', 'detik', 'liputan6', 'antaranews', 'republika', 'bisnis', 'cnbc', 'tirto', 'katadata', 'kumparan', 'merdeka', 'tribun', 'jawapos', 'suara']
CREDIBLE_MED = ['viva', 'okezone', 'sindonews', 'grid', 'jpnn']
SOCIAL_MEDIA = ['facebook', 'twitter', 'tiktok', 'instagram', 'whatsapp', 'telegram', 'youtube', 'blog', 'wordpress']
FACT_CHECK_SITES = ['turnbackhoax.id', 'kompas.com/cekfakta', 'cekfakta.com', 'liputan6.com/cek-fakta', 'antaranews.com', 'tempo.co']

def live_fact_check(title):
    if not title:
        return [], 0.0

    query = f"{title} cek fakta hoax"
    references = []
    prob_adjustment = 0.0

    try:
        results = DDGS().text(query, region='id-id', safesearch='moderate', max_results=3)
        for r in results:
            ref_url = r.get('href', '').lower()
            snippet = r.get('body', '').lower()
            ref_title = r.get('title', '')
            
            references.append({
                'title': ref_title,
                'url': r.get('href', ''),
                'snippet': r.get('body', '')
            })

            # Check if from trusted fact-checking site
            if any(site in ref_url for site in FACT_CHECK_SITES):
                # Analyze snippet for hoax indicators
                if any(word in snippet for word in ['hoaks', 'salah', 'keliru', 'disinformasi', 'palsu', 'manipulasi', 'hoax']):
                    prob_adjustment += 0.5 # Strong hoax signal
                elif any(word in snippet for word in ['fakta', 'benar', 'valid', 'terbukti']):
                    prob_adjustment -= 0.5 # Strong fact signal
    except Exception as e:
        print(f"DuckDuckGo Search error: {e}")
        pass
        
    return references, prob_adjustment

def check_hyperbolic_title(title):
    if not title:
        return 0.0
        
    penalty = 0.0
    title_upper = title.upper()
    title_lower = title.lower()
    
    # 1. Excessive punctuation
    if '!!!' in title or '??' in title or '!?' in title or '?!' in title:
        penalty += 0.2
        
    # 2. Sensational/Hyperbolic keywords
    sensational_words = ['viral', 'gempar', 'terbongkar', 'ngeri', 'mengerikan', 'kiamat', 'konspirasi', 'azab', 'heboh', 'waspada', 'mengejutkan', 'bocor', 'rahasia', 'wajib baca', 'gila', 'gempar!!!']
    for word in sensational_words:
        if word in title_lower:
            penalty += 0.15
            break
            
    # 3. Excessive ALL CAPS words (indicative of clickbait/hoax)
    words = title.split()
    if len(words) > 3:
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 3)
        if all_caps_words >= 3:
            penalty += 0.15
            
    return min(0.5, penalty) # Cap penalty at 0.5 (very hyperbolic)

def adjust_probability(prob_hoax, publisher, title):
    # 1. Evaluasi kredibilitas media
    prior_hoax = 0.5
    weight = 0.0
    
    if publisher:
        publisher = publisher.lower().strip()
        if any(m in publisher for m in CREDIBLE_HIGH):
            prior_hoax = 0.05
            weight = 0.8  # Sangat percaya pada media jurnalistik resmi
        elif any(m in publisher for m in CREDIBLE_MED):
            prior_hoax = 0.2
            weight = 0.5
        elif any(m in publisher for m in SOCIAL_MEDIA):
            prior_hoax = 0.8
            weight = 0.4

    adjusted_prob = (1 - weight) * prob_hoax + weight * prior_hoax
    
    # 2. Evaluasi Diksi Judul (Penalty untuk judul hiperbola)
    hyperbolic_penalty = check_hyperbolic_title(title)
    if hyperbolic_penalty > 0:
        adjusted_prob = adjusted_prob + (hyperbolic_penalty * (1 - adjusted_prob))

    return max(0.01, min(0.99, adjusted_prob))

MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 300

class HoaxLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, drop_prob=0.3):
        super(HoaxLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

# Load Model and Tokenizer
MODEL_PATH = 'hoax_lstm_model.pth'
TOKENIZER_PATH = 'word2idx.pickle'

model = None
word2idx = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, 'rb') as handle:
        word2idx = pickle.load(handle)
        
    print("Loading model...")
    model = HoaxLSTM(
        vocab_size=MAX_VOCAB_SIZE, 
        embed_dim=100, 
        hidden_dim=64, 
        output_dim=1, 
        n_layers=2
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model and Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def text_to_sequence(text, word2idx, max_len):
    words = text.split()
    seq = [word2idx.get(w, 1) for w in words]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or word2idx is None:
        return jsonify({'error': 'Model or Tokenizer not loaded on server. Please train the model first.'}), 500

    data = request.json
    url_str = data.get('url', '')

    if not url_str.strip():
        return jsonify({'error': 'Please provide a URL to analyze.'}), 400

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url_str, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title_tag = soup.find('title')
        title = title_tag.text.strip() if title_tag else ""
        
        parsed_url = urlparse(url_str)
        domain = parsed_url.hostname.replace('www.', '') if parsed_url.hostname else ""
        publisher = domain.split('.')[0] if domain else ""
        
        paragraphs = soup.find_all('p')
        text = " ".join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 10])
        
        if not text:
            text = title
            
    except Exception as e:
        return jsonify({'error': f'Gagal mengekstraksi konten dari URL: {str(e)}'}), 400

    # Gabungkan judul dan teks untuk input model
    full_text = f"{title} {text}".strip()

    cleaned = clean_text(full_text)
    seq = text_to_sequence(cleaned, word2idx, MAX_SEQUENCE_LENGTH)
    
    tensor_input = torch.tensor([seq], dtype=torch.long).to(device)
    
    with torch.no_grad():
        prediction = model(tensor_input)
        raw_prob_hoax = float(prediction.item())
        
    # Adjust based on credibility and title diction
    prob_hoax = adjust_probability(raw_prob_hoax, publisher, title)
    
    # Live Fact-Checking
    references, fact_check_adj = live_fact_check(title)
    prob_hoax = prob_hoax + fact_check_adj
    prob_hoax = max(0.01, min(0.99, prob_hoax))
    
    # Label 1 = Hoax, 0 = Fact
    is_hoax = prob_hoax > 0.5

    result = {
        'prediction': 'HOAX' if is_hoax else 'FAKTA',
        'confidence': prob_hoax if is_hoax else 1.0 - prob_hoax,
        'prob_hoax': prob_hoax,
        'raw_prob': raw_prob_hoax,
        'extracted_title': title,
        'extracted_publisher': publisher,
        'extracted_text': text[:600] + "..." if len(text) > 600 else text,
        'references': references
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
