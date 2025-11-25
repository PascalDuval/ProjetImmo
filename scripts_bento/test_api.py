import requests

url = "http://127.0.0.1:3000/predict"

data = {
    "data": {  # 👈 nécessaire avec BentoML 1.4.29
        "BuildingAge": 10,
        "log_surface": 3.6,
        "has_parking": 1,
        "Use_Office": 0,
        "Use_Other": 0,
        "Use_Retail": 1,
        "Use_Warehouse": 0,
        "Use_Unknown": 0
    }
}

response = requests.post(url, json=data)
print("✅ Status:", response.status_code)
print("📦 Response:", response.json())
