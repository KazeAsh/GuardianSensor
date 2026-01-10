import pandas as pd
import requests
import cv2
import os
from datetime import datetime
import random

class DataCollector:
    def __init__(self):
        self.raw_data = "data/raw"
        os.makedirs(self.raw_data, exist_ok=True)

    # Collect real data from weather in Japan
    def collect_weather_data(self, city="Tokyo"):
        # Using OpenWeatherMap API (free version)
       
        api_key = "API_WEATHER"  # Replace with your actual API key
        if api_key == "API_WEATHER":
            print("Warning: Please set your OpenWeatherMap API key in data_collector.py")
            print("Get a free API key from: https://openweathermap.org/api")
            return self.generate_mock_weather_data(city)
        
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Save as a CSV file
                df = pd.DataFrame([{
                    "city": city,
                    "timestamp": datetime.now(),
                    "temperature_c": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "weather": data["weather"][0]["description"],
                    "wind_speed": data["wind"]["speed"],
                    "source": "OpenWeatherMap"
                }])
                
                file_path = os.path.join(self.raw_data, f"weather_{city}.csv")
                df.to_csv(file_path, index=False)
                print(f"Weather data for {city} collected successfully.")
                return df
            else:
                print(f"Failed to get weather data. Status code: {response.status_code}")
                return self.generate_mock_weather_data(city)
                
        except Exception as e:
            print(f"Error collecting weather data: {e}")
            return self.generate_mock_weather_data(city)
    
    def generate_mock_weather_data(self, city="Tokyo"):
        """Generate mock weather data when API is not available"""
        print(f"Generating mock weather data for {city}")
        
        df = pd.DataFrame([{
            "city": city,
            "timestamp": datetime.now(),
            "temperature_c": random.uniform(15, 30),
            "humidity": random.uniform(40, 80),
            "pressure": random.uniform(1000, 1020),
            "weather": random.choice(["clear", "cloudy", "rainy", "sunny"]),
            "wind_speed": random.uniform(0, 10),
            "source": "MockData"
        }])
        
        file_path = os.path.join(self.raw_data, f"weather_{city}.csv")
        df.to_csv(file_path, index=False)
        return df

if __name__ == "__main__":
    collector = DataCollector()
    weather_data = collector.collect_weather_data("Tokyo")
    
    if weather_data is not None:
        print(f"\nData collection successful:")
        print(weather_data)
        print(f"\nData saved to: data/raw/weather_Tokyo.csv")