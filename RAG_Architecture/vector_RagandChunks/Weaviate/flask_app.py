from flask import Flask, request, jsonify
import threading
import json
import numpy as np
import torch
import threading
import logging
import os
from utils import generate_embedding

# Lazy-loaded reranker. We avoid importing/initializing heavy model
# at module import time to prevent circular-imports or missing-dependency
# errors when the notebook only wants to import the module.
reranker = None

def init_reranker():
    """Initialize the FlagReranker on first use.

    Returns the instantiated reranker or raises an informative exception.
    """
    global reranker
    if reranker is not None:
        return reranker

    try:
        # Import local model wrapper only when needed
        from FlagEmbedding import FlagReranker
    except Exception as e:
        raise ImportError(
            "FlagEmbedding module is not available in the environment. "
            "Install or provide the module before calling /rerank. "
            f"Original error: {e}"
        )

    model_cache = os.environ.get("MODEL_M3")
    if not model_cache:
        raise EnvironmentError(
            "Environment variable MODEL_M3 is not set. Set MODEL_M3 to a local cache path for models."
        )

    # instantiate the reranker
    reranker = FlagReranker('BAAI/bge-reranker-base', cache_dir=model_cache, use_fp16=False)
    return reranker

app = Flask(__name__)

@app.route('/.well-known/ready', methods=['GET'])
def readiness_check():
    return "Ready", 200

@app.route('/meta', methods=['GET'])
def readiness_check_2():
    return jsonify({'status': 'Ready'}), 200

@app.route('/rerank', methods=['POST'])
def rerank():
    try:
        data = None
        try:
            # Attempt to parse as JSON first
            data = request.json
            if data is None:
                # If request.json was empty, try decoding raw data as JSON string
                text_str = request.data.decode("utf-8")
                data = json.loads(text_str)
            # The entire request body is the JSON object Weaviate sends
            text = data
        except Exception as e:
            # Fallback for unexpected data formats
            try:
                text_str = request.data.decode("utf-8")
                text = json.loads(text_str)
            except Exception as e_inner:
                print(f"Error parsing request data: {e_inner}")
                return jsonify({'error': f"Could not parse request body: {e_inner}"}), 400

        # Validate expected input format from Weaviate
        if not isinstance(text, dict) or 'query' not in text or 'documents' not in text:
            print(f"Invalid input format. Expected dict with 'query' and 'documents'. Got: {text}")
            return jsonify({'error': "Invalid input format. Expected a dictionary with 'query' and 'documents'."}), 400

        query = text['query']
        documents = text['documents']

        if not documents:
            # Return an empty list of scores if no documents are provided for reranking
            # This handles cases where Weaviate might send an empty list, preventing errors
            return jsonify({'scores': []})

        # Ensure reranker is initialized (lazy init)
        try:
            rr = init_reranker()
        except Exception as init_err:
            msg = f"Failed to initialize reranker: {init_err}"
            print(msg)
            return jsonify({'error': msg}), 500

        # Prepare pairs for the reranker model
        compares = [(query, doc) for doc in documents]

        # Compute scores using the FlagReranker model
        scores = rr.compute_score(compares)

        # Convert scores (typically a NumPy array or tensor) to a Python list
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else scores

        # Construct the response in the format Weaviate's reranker-transformers module expects
        # This includes the original document text and its computed score
        reranked_results = []
        for i, doc_text in enumerate(documents):
            score = scores_list[i]
            reranked_results.append({
                "document": doc_text,  # Include the original document text
                "score": float(score)  # Use "score" as the key, ensuring float type
            })

        return jsonify({'scores': reranked_results}) # Top-level key is "scores" (plural)

    except Exception as e:
        print(f"Unhandled error in /rerank: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/vectors', methods=['POST']) 
def vectorize():
    try:
        try:
            data = request.json.get('text')
        except Exception as e:
            try:
                data = request.data.decode("utf-8")
            except Exception as e:
                print(e)
        text = json.loads(data)
        if isinstance(text, str):
            text = [text]
        else:
            text =text['text']
            
        embeddings = generate_embedding(text)

        return jsonify({'vector': embeddings})


    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
app.logger.disabled = True
# Get the Flask app's logger
log = logging.getLogger('werkzeug')
# Set logging level (ERROR or CRITICAL suppresses routing logs)
log.setLevel(logging.ERROR)
def run_app():
    port = int(os.environ.get('FLASK_PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug = False)

if __name__ == "__main__":
    # If executed directly, start the server in the foreground.
    run_app()
else:
    # When imported (e.g., from a notebook), start the server in a background thread.
    try:
        flask_thread = threading.Thread(target=run_app, daemon=True)
        flask_thread.start()
    except Exception as e:
        # Log and continue; notebooks can still call run_app() explicitly if needed.
        print(f"Could not start flask thread automatically: {e}")
