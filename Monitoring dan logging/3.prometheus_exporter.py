import time
import psutil
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import threading
import random

app = Flask(__name__)

# === 1. Metrics Definition (10 Metrics for Advance) ===

# A. Request Metrics
REQUEST_COUNT = Counter('request_count_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds')
ERROR_COUNT = Counter('error_count_total', 'Total number of errors')

# B. System Metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Current CPU usage percent')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Current RAM usage in bytes')

# C. Model Metrics (Prediction)
PRED_CLASS_YES = Counter('prediction_class_yes_total', 'Total "yes" predictions')
PRED_CLASS_NO = Counter('prediction_class_no_total', 'Total "no" predictions')
PRED_CONFIDENCE = Histogram('prediction_confidence_score', 'Model confidence score distribution')

# D. Data Drift / Input Metrics
INPUT_AGE_MEAN = Gauge('input_age_mean', 'Moving average of input age')
INPUT_CAMPAIGN_MEAN = Gauge('input_campaign_mean', 'Moving average of campaign contacts')

# Helper for system metrics
def update_system_metrics():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        time.sleep(5)

# Start system monitoring in background
sys_thread = threading.Thread(target=update_system_metrics)
sys_thread.daemon = True
sys_thread.start()

# === 2. Mock Model Loading (Simulated for Monitoring Demo) ===
# In real production, we would load the MLflow model here.
# clf = mlflow.pyfunc.load_model("models:/...")

def mock_predict(data):
    # Simulate prediction logic
    # Bank Marketing logic simulation
    confidence = random.uniform(0.1, 0.9)
    # If duration > 500 (just a heuristic for simulation), likely yes
    # But duration is dropped. Let's use 'previous'.
    
    # Randomize generic prediction
    pred = 1 if confidence > 0.7 else 0
    
    return pred, confidence

# === 3. API Endpoints ===

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.inc()
    try:
        data = request.json
        if not data:
            ERROR_COUNT.inc()
            return jsonify({'error': 'No data provided'}), 400
        
        # Simulate Processing
        age = data.get('age', 30)
        campaign = data.get('campaign', 1)
        
        # Update Input Metrics
        INPUT_AGE_MEAN.set(age) # In real app, calculate rolling avg
        INPUT_CAMPAIGN_MEAN.set(campaign)
        
        # Predict
        pred, conf = mock_predict(data)
        
        if pred == 1:
            PRED_CLASS_YES.inc()
        else:
            PRED_CLASS_NO.inc()
            
        PRED_CONFIDENCE.observe(conf)
        
        # Simulate occasional error for alerting testing
        if random.random() < 0.05: # 5% chance of error
            raise Exception("Simulated Internal Error")
            
        # Simulate high latency occasionally
        if random.random() < 0.05:
            time.sleep(2.5) # Trigger High Latency Alert

        result = {'prediction': 'yes' if pred == 1 else 'no', 'confidence': conf}
        return jsonify(result)
        
    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return "OK"

if __name__ == '__main__':
    # Start Prometheus Metrics Server on a different port if needed, 
    # but flask_prometheus_exporter usually exposes on /metrics.
    # Here we use prometheus_client to expose on /metrics manually is trickier with pure Flask 
    # unless using DispatcherMiddleware. 
    # Simpler: Use prometheus_client's start_http_server for metrics, and Flask for app.
    # But usually we want them on same port for simplicity in 'localhost:5000'.
    
    # We will simply add a /metrics route that returns the registry
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    @app.route('/metrics')
    def metrics():
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

    print("Starting App on port 5000...")
    app.run(host='0.0.0.0', port=5000)
