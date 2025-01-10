import requests

url = "http://127.0.0.1:9696/predict"

trip = {
    "PULocationID": 100,
    "DOLocationID": 102,
    "trip_distance": 20
}

response = requests.post(url, json=trip)
print(response.json())
