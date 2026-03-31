import time
import random
import requests

def simulate_time_travel():
    print("==========================================================")
    print("🚀 INITIALIZING TIME TRAVEL SIMULATION")
    print("   Simulating 6 Months of Macroeconomic Shock in Kigali...")
    print("==========================================================\n")
    
    API_URL = "http://localhost:8000/feedback"

    for month in range(1, 7):
        print(f"🌍 Month {month}: Economic conditions tightening...")
        
        # In later months, income drops, credit requests go up, default rate spikes
        income_modifier = 1.0 - (month * 0.1)  # Income drops by up to 60%
        default_prob_base = 0.1 + (month * 0.12) # Base default probability rises
        
        success_count = 0
        for i in range(20): # 20 synthetic loans per month = 120 total
            base_income = random.uniform(30000, 300000)
            
            payload = {
                "AMT_INCOME_TOTAL": base_income * income_modifier,
                "AMT_CREDIT": random.uniform(100000, 1000000),
                "AMT_ANNUITY": random.uniform(5000, 50000),
                "AMT_GOODS_PRICE": random.uniform(100000, 1000000),
                "DAYS_BIRTH": -random.randint(7000, 20000),
                "DAYS_EMPLOYED": -random.randint(100, 10000),
                "CNT_CHILDREN": random.randint(0, 5),
                "CNT_FAM_MEMBERS": random.randint(1, 8),
                "EXT_SOURCE_1": random.choice([0.08, 0.35, 0.55, 0.75]),
                "EXT_SOURCE_2": random.uniform(0.1, 0.8),
                "EXT_SOURCE_3": random.uniform(0.1, 0.8),
                "FLAG_OWN_CAR": random.choice([0, 1]),
                "CODE_GENDER": random.choice(["M", "F"]),
                "NAME_EDUCATION_TYPE": random.choice(["Higher education", "Secondary / secondary special"]),
                "NAME_INCOME_TYPE": random.choice(["Working", "State servant", "Commercial associate"]),
                "NAME_HOUSING_TYPE": random.choice(["House / apartment", "Rented apartment", "With parents"]),
                "prediction_prob": random.uniform(0.0, 1.0),
                "ground_truth": 1 if random.random() < default_prob_base else 0
            }
            
            # Intentionally spike prediction_prob in later months to trigger KS drift
            if month > 3:
                payload["prediction_prob"] = min(1.0, payload["prediction_prob"] + 0.3)
                
            try:
                resp = requests.post(API_URL, json=payload, timeout=2)
                if resp.status_code == 200:
                    success_count += 1
            except Exception as e:
                print(f"   [!] Connection failed. Is your FastAPI server (start_services.py) running? {e}")
                return
                
        print(f"   ↳ {success_count} synthetic loan resolutions recorded in database.")
        time.sleep(1.5) # Dramatic pause for the audience
        
    print("\n✅ TIME TRAVEL COMPLETE!")
    print("   120 stressed market loan outcomes have been securely recorded into the backend.")
    print("👉 Go to your Streamlit 'MLOps Control Center' and click 'Run Drift Monitor Batch Scan' to see the alarms trigger!")

if __name__ == "__main__":
    simulate_time_travel()
