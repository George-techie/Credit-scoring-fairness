import time
import random
import requests
import subprocess
import sys

API_URL = "http://localhost:8000/feedback"

def run_mlops_history_simulation():
    print("==========================================================")
    print("📈 INITIALIZING MLOPS HISTORY GENERATION")
    print("   Simulating 8 Continuous Loops of Data Collection, Drift Detection, and Retraining...")
    print("==========================================================\n")
    
    for loop in range(1, 9):
        print(f"\n🌀 Starting Validation Cycle {loop}/8...")
        
        # 1. Inject Data
        success_count = 0
        default_prob = 0.1 + (loop * 0.05) # increasing drift logic
        for i in range(150): # 150 rows guarantees retrain pipeline thresholds
            base_income = random.uniform(50000, 300000)
            payload = {
                "AMT_INCOME_TOTAL": base_income,
                "AMT_CREDIT": random.uniform(100000, 800000),
                "AMT_ANNUITY": random.uniform(5000, 40000),
                "AMT_GOODS_PRICE": random.uniform(100000, 800000),
                "DAYS_BIRTH": -random.randint(7000, 20000),
                "DAYS_EMPLOYED": -random.randint(100, 10000),
                "CNT_CHILDREN": random.randint(0, 4),
                "CNT_FAM_MEMBERS": random.randint(1, 6),
                "EXT_SOURCE_1": random.choice([0.1, 0.4, 0.6, 0.8]),
                "EXT_SOURCE_2": random.uniform(0.1, 0.8),
                "EXT_SOURCE_3": random.uniform(0.1, 0.8),
                "FLAG_OWN_CAR": random.choice([0, 1]),
                "CODE_GENDER": random.choice(["M", "F"]),
                "NAME_EDUCATION_TYPE": random.choice(["Higher education", "Secondary / secondary special"]),
                "NAME_INCOME_TYPE": random.choice(["Working", "State servant", "Commercial associate"]),
                "NAME_HOUSING_TYPE": random.choice(["House / apartment", "Rented apartment", "With parents"]),
                "prediction_prob": min(1.0, random.uniform(0.1, 0.8) + (loop * 0.05)),
                "ground_truth": 1 if random.random() < default_prob else 0
            }
            try:
                resp = requests.post(API_URL, json=payload, timeout=2)
                if resp.status_code == 200:
                    success_count += 1
            except Exception as e:
                print(f"   [!] Connection failed. Is FastAPI running on port 8000? {e}")
                return
                
        print(f"   ↳ {success_count} synthetic ground-truth outcomes recorded.")
        time.sleep(1)
        
        # 2. Run Drift Monitor
        print("   ↳ Running Drift Monitor (monitor_drift.py)...")
        subprocess.run([sys.executable, "monitor_drift.py"], capture_output=True)
        time.sleep(1)
        
        # 3. Trigger Retraining
        print("   ↳ Triggering Continuous Retraining Pipeline (retrain_pipeline.py)...")
        subprocess.run([sys.executable, "retrain_pipeline.py"], capture_output=True)
        time.sleep(1)
        
    print("\n✅ Deep MLflow History Successfully Generated!")
    print("   Open your Streamlit MLOps Control Center tab. You will now see 8 dynamic historical points plotted perfectly over time!")

if __name__ == "__main__":
    run_mlops_history_simulation()
