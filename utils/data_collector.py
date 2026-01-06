import pandas as pd
import requests
import cv2
import os

class DataCollector:
    def __init__(self):
        self.raw_data = "data/raw"
        os.makedirs(self.raw_data, exist_ok=True)

    #collect read data from weather in Japan
    def collect_weather_data(self, city="Tokyo"):
        # Using OpenWeatherMap API (free version)
        api_key = "codehere" #openweathermap.org
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp_kelvin = data["main"]["temp"] 
            temp_celsius = temp_kelvin - 273.15

            # Save as a CSV file
            df = pd.DataFrame([{
                "city": city,
                "Time Stamp": pd.Timestamp.now(),
                "temperature_celsius": temp_celsius,
                "humidity": data["main"]["humidity"],
                "weather": data["weather"][0]["description"],
                "Source": "OpenWeatherMap"
            }])
            df.to_csv(os.path.join(self.raw_data, f"weather_{city}.csv"), index=False)
            print(f"Weather data for {city} collected successfully.")

            return df
        return None
    
if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_weather_data("Tokyo")

