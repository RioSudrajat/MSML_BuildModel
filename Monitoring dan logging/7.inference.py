import requests
import time
import random

# URL Application
url = 'http://localhost:5000/predict'

# Dummy Data Generator
def generate_dummy_data():
    return {
        "age": random.randint(20, 60),
        "job": random.choice(["admin", "technician", "blue-collar"]),
        "marital": random.choice(["married", "single", "divorced"]),
        "education": "university.degree",
        "default": "no",
        "housing": random.choice(["yes", "no"]),
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": random.randint(1, 10),
        "pdays": 999,
        "previous": random.randint(0, 2),
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.893,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.299,
        "nr.employed": 5099.1
    }

print("Starting Traffic Generator for Monitoring...")
print("Press Ctrl+C to stop.")

try:
    while True:
        data = generate_dummy_data()
        try:
            response = requests.post(url, json=data)
            print(f"Status: {response.status_code}, Res: {response.json()}")
        except Exception as e:
            print(f"Request failed: {e}")
        
        # Random sleep to vary traffic load
        time.sleep(random.uniform(0.1, 1.0))

except KeyboardInterrupt:
    print("Traffic Generator Stopped.")
