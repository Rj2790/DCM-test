import requests
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
from dotenv import load_dotenv  

load_dotenv()

app = Flask(__name__)
CORS(app)

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise Exception("OPENROUTER_API_KEY environment variable not set")

# Get Google Sheets webhook URL from environment variable
GOOGLE_SHEETS_WEBHOOK_URL = os.getenv('GOOGLE_SHEETS_WEBHOOK_URL')
if not GOOGLE_SHEETS_WEBHOOK_URL:
    print("Warning: GOOGLE_SHEETS_WEBHOOK_URL not set - Google Sheets integration disabled")

# Initialize sentence transformer model
print("Loading sentence transformer model...")
model = SentenceTransformer('BAAI/bge-small-en')
print("Model loaded successfully!")

# Load precomputed FAQ embeddings
print("Loading precomputed FAQ embeddings...")
with open("faq_embeddings.json", "r") as f:
    FAQS = json.load(f)
print(f"Loaded {len(FAQS)} FAQ embeddings!")

def get_embedding(text):
    """Get embedding using sentence-transformers"""
    try:
        embedding = model.encode([text])[0]
        return embedding.tolist()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_inappropriate_query(query):
    """Check if query contains inappropriate content"""
    inappropriate_patterns = [
        r'\b(fuck|shit|bitch|ass|dick|pussy|boobs|tits)\b',
        r'\b(fuck\s+you|fuck\s+off|go\s+fuck)\b',
        r'\b(send\s+me\s+your\s+boobs|nudes|pictures)\b',
        r'\b(kill\s+yourself|die|suicide)\b',
        r'\b(hate\s+you|hate\s+this|sucks)\b'
    ]
    
    query_lower = query.lower()
    for pattern in inappropriate_patterns:
        if re.search(pattern, query_lower):
            return True
    return False

def get_llm_response(user_query, faq_answer):
    """Use OpenRouter LLM to make the answer more natural"""
    try:
        prompt = f"""You are a professional customer service representative for DCM Moguls, a digital marketing company.

User asked: "{user_query}"

Here is the relevant information: {faq_answer}

Please provide a concise, professional, and polite response that directly answers the user's question. Keep it brief (2-3 sentences maximum) and maintain a professional yet friendly tone. Be straightforward but courteous.

IMPORTANT: Do NOT include any preamble, explanation, or phrases like "Here's a possible response." Do NOT wrap your response in quotes. Only reply with the message you would send to the user.
"""
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://huggingface.co",
                "X-Title": "DCM Moguls Chatbot",
            },
            data=json.dumps({
                "model": "meta-llama/llama-3.2-1b-instruct:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3
            })
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result['choices'][0]['message']['content'].strip()
            
            # Remove quotes from beginning and end if present
            if llm_response.startswith('"') and llm_response.endswith('"'):
                llm_response = llm_response[1:-1]
            
            return llm_response
        else:
            print(f"LLM API error: {response.status_code} - {response.text}")
            return faq_answer
            
    except Exception as e:
        print(f"LLM error: {e}")
        return faq_answer

def send_to_google_sheets(user_data):
    """Send user data to Google Sheets webhook"""
    if not GOOGLE_SHEETS_WEBHOOK_URL:
        print("Google Sheets webhook URL not configured")
        return False
    
    payload = {
        "name": user_data.get("fullName", ""),
        "email": user_data.get("email", ""),
        "phone": user_data.get("phone", ""),
        "professionType": user_data.get("businessType", "")
    }
    
    try:
        response = requests.post(GOOGLE_SHEETS_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            print("Data sent to Google Sheets successfully")
            return True
        else:
            print(f"Failed to send to Google Sheets: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error sending to Google Sheets: {e}")
        return False

@app.route("/")
def home():
    return send_from_directory('.', 'web.html')

@app.route("/faq", methods=["POST"])
def faq():
    user_query = request.json["query"]
    print("Received query:", user_query)
    
    # Check for inappropriate content first
    if is_inappropriate_query(user_query):
        print("Inappropriate query detected - returning fixed response")
        return jsonify({
            "answer": "I'm sorry, I won't be able to help you with that. Do you have any other queries about our company and the services we provide?"
        })
    
    # Get embedding for user query
    user_embedding = get_embedding(user_query)
    if user_embedding is None:
        return jsonify({"answer": "Sorry, there was an error processing your question."}), 500

    # Compute similarity with each FAQ
    similarities = []
    for faq in FAQS:
        if faq["embedding"] is not None:
            similarity = cosine_similarity(user_embedding, faq["embedding"])
            similarities.append(similarity)
        else:
            similarities.append(0)
    
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]
    
    print(f"Best match: {FAQS[best_idx]['question']} (similarity: {best_score:.3f})")

    if best_score > 0.8:  
        faq_answer = FAQS[best_idx]["answer"]
        
        # Use LLM to make the answer more natural
        print("Using LLM to generate natural response...")
        natural_answer = get_llm_response(user_query, faq_answer)
        print(f"LLM response: {natural_answer}")
        
        answer = natural_answer
    else:
        print(f"Similarity score {best_score:.3f} below threshold 0.8 - returning sorry message")
        answer = "I'm sorry, I couldn't find an answer to your question. Do you have any other queries about our company and the services we provide?"

    return jsonify({"answer": answer})

@app.route("/save-user", methods=["POST"])
def save_user():
    user_data = request.json
    success = send_to_google_sheets(user_data)
    
    if success:
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))