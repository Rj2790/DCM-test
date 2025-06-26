import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
import requests
import json
import numpy as np
import re

app = Flask(__name__)

# Simplified version without sentence-transformers for now
def simple_faq_response(user_query):
    """Simple FAQ response without ML"""
    faq_data = {
        "pricing": "Our pricing is customized based on your specific needs. We offer transparent, upfront pricing with no hidden fees. Most clients see a 600%+ ROI within the first 6 months.",
        "services": "We offer comprehensive digital marketing services including SEO, PPC, social media marketing, and content creation.",
        "contact": "You can contact us through our website or schedule a free consultation to discuss your needs."
    }
    
    query_lower = user_query.lower()
    
    if "price" in query_lower or "cost" in query_lower:
        return faq_data["pricing"]
    elif "service" in query_lower or "offer" in query_lower:
        return faq_data["services"]
    else:
        return faq_data["contact"]

@app.route("/faq", methods=["POST"])
def faq():
    user_query = request.json["query"]
    answer = simple_faq_response(user_query)
    return jsonify({"answer": answer})

@app.route("/save-user", methods=["POST"])
def save_user():
    user_data = request.json
    # Implement Google Sheets integration here
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run()
