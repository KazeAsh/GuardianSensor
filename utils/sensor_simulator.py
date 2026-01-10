import random
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Simulate sensor data for child safety monitoring
class SensorSimulator:

    # Simulate One hour of sensor data
    def simulate_one_hour(self):
        data_points = []

        # Base temperature
        base_temp = random.uniform(18, 24)  # Comfortable temperature in Celsius

        for minute in range(60):
            # Start time
            timestamp = datetime.now() + timedelta(minutes=minute)

            # Scenario: car parked for 10 minutes, child left inside for 20 minutes, then car starts moving again
            if minute < 10:  # Driving with both adult and child
                door_state = "closed"
                engine_state = "on"
                weight_sensor_left = 15.0  # Weight of child in kg
                weight_sensor_right = 75.0  # Weight of adult in kg
            elif 10 <= minute < 30:  # Parked with child inside, adult exited
                door_state = "closed"
                engine_state = "off"
                weight_sensor_left = 15.0
                weight_sensor_right = 0.0
            else:  # Car starts moving again, adult returns
                door_state = "closed"
                engine_state = "on"
                weight_sensor_left = 15.0
                weight_right_kg = 75.0  # Fix: Changed variable name
            
            # Simulate temperature changes - MORE REALISTIC
            if engine_state == "off" and minute >= 10:
                # Car is parked and heats up - now with more realistic pattern
                # First 10 minutes: slow rise, next 10: faster rise
                parked_time = minute - 10
                if parked_time < 10:
                    temp_increase = parked_time * 0.3  # 0.3°C per minute
                else:
                    temp_increase = 3 + (parked_time - 10) * 0.7  # 0.7°C per minute after 10 minutes
                
                # Add some random spikes for realism
                if random.random() < 0.1:  # 10% chance of a temperature spike
                    spike = random.uniform(2, 5)
                    temp_increase += spike
                
                current_temp = base_temp + min(temp_increase, 15)  # Cap at +15°C
                current_temp += random.uniform(-0.5, 0.5)  # Small random variation
            else:
                # Car is running, maintain base temp with occasional variations
                if random.random() < 0.05:  # 5% chance of AC malfunction
                    current_temp = base_temp + random.uniform(3, 8)
                else:
                    current_temp = base_temp + random.uniform(-1.5, 1.5)

            # Motion detection (child might move occasionally)
            motion_detected = random.random() > 0.7 if minute > 10 else True
            
            # Car data points of inside temperature, door state, engine state, weight sensors, motion detection
            data_point = {
                'timestamp': timestamp,
                'engine_state': engine_state,
                'door_state': door_state,
                'temperature_c': round(current_temp, 2),
                'weight_left_kg': weight_sensor_left,
                'weight_right_kg': weight_sensor_right,  # Fix: Changed variable name
                'motion_detected': motion_detected,
                'co2_level': random.uniform(400, 1200),  # ppm - wider range
                'humidity': random.uniform(30, 80)  # percent - wider range
            }
            
            data_points.append(data_point)
        
        df = pd.DataFrame(data_points)
        
        # Create directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/sensor_data_hour.csv", index=False)
        print(f"Generated {len(df)} sensor readings")
        
        # Print temperature statistics
        print(f"Temperature range: {df['temperature_c'].min():.1f}°C to {df['temperature_c'].max():.1f}°C")
        print(f"Temperature mean: {df['temperature_c'].mean():.1f}°C")
        
        return df


if __name__ == "__main__":
    simulator = SensorSimulator()
    df = simulator.simulate_one_hour()