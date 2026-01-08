import requests
import json
from datetime import datetime, timedelta

# Create a client to interact with a GardianSeat API
class GuardianSeatClient:
    def __init__(self, base_url = "https://localhost:8000"):
      self.base_url = base_url

    # Send data to the API
    def send_sensor_data(self, data):
        url = f"{self.base_url}/sensor-data"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        return response.json()