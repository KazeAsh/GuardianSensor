import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
import json
from scipy import stats

# Clean and validate data for GardianSeat project
class DataCleaningPipeline:

    def __init__(self):
        self.raw_data = "data/raw"
        self.processed_path = "data/processed"
        os.makedirs(self.processed_path, exist_ok=True)

        # Define validation rules
        self.validation_rules = {
            'temperature_c': {'min': -30, 'max': 80, 'required': True},
            'weight_left_kg': {'min': 0, 'max': 100, 'required': True},
            'weight_right_kg': {'min': 0, 'max': 100, 'required': True},
            'co2_level': {'min': 300, 'max': 5000, 'required': False},
            'humidity': {'min': 0, 'max': 100, 'required': False}
        }
        
        # Domain-specific thresholds for child safety
        self.safety_thresholds = {
            'temperature_c': {'danger': 40, 'warning': 26},  # °C
            'co2_level': {'danger': 2000, 'warning': 1500},  # ppm
            'humidity': {'danger': 85, 'warning': 70},  # %
            'max_parked_time': 30,  # minutes
        }

    # Full cleaning pipeline
    def clean_sensor_data(self, file_path: str) -> pd.DataFrame:
        
        # Display file being processed
        print(f"Cleaning data from {file_path}")

        # Load raw data
        df = pd.read_csv(file_path)
        original_count = len(df)
        print(f"Loaded original data count records: {original_count}")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # handle missing values
        df_cleaned = self.handle_missing_values(df)

        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates(subset=['timestamp'], keep='first')
        print(f"Data count after removing duplicates: {len(df_cleaned)}")

        # Validate data range
        df_cleaned = self.validate_data_ranges(df_cleaned)

        # Detect and handle outliers with DOMAIN-SPECIFIC methods
        df_cleaned, outlier_report = self.handle_outliers_domain_specific(df_cleaned)
        
        # Create derived features
        df_cleaned = self.create_features(df_cleaned)

        # Save cleaned data
        self.save_cleaning_report(df, df_cleaned, outlier_report, file_path)

        # Print column value analysis to debug constant columns
        print("\nColumn value analysis:")
        for col in df_cleaned.columns:
            unique_values = df_cleaned[col].nunique()
            if unique_values == 1:
                print(f"  {col}: Constant value = {df_cleaned[col].iloc[0]}")
            elif unique_values < 5:
                print(f"  {col}: Only {unique_values} unique values")

        cleaned_count = len(df_cleaned)
        print(f"  Cleaning complete: {original_count} → {cleaned_count} records "
              f"({((original_count-cleaned_count)/original_count*100):.1f}% removed)")
        
        return df_cleaned
    
    # Create strategies to handle missing values in different columns
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()

        # For numerical columns, use interpolation for time series data
        num_cols = ['temperature_c', 'weight_left_kg', 'weight_right_kg', 'co2_level', 'humidity']
        for col in num_cols:
            if col in df_filled.columns:
                # Use linear interpolation for time series
                df_filled[col] = df_filled[col].interpolate(method='linear', limit_direction='both')
                
                # Fill any remaining NaN with median
                if df_filled[col].isna().any():
                    median_val = df_filled[col].median()
                    df_filled[col] = df_filled[col].fillna(median_val)
            
        # For categorical, use forward fill for time series
        categorical_cols = ['engine_state', 'door_state']
        for col in categorical_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].ffill().bfill()
                # Fill any remaining with most common value
                if df_filled[col].isna().any():
                    mode_val = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'unknown'
                    df_filled[col] = df_filled[col].fillna(mode_val)
        
        # Also handle motion_detected column
        if 'motion_detected' in df_filled.columns:
            df_filled['motion_detected'] = df_filled['motion_detected'].ffill().bfill()
            if df_filled['motion_detected'].isna().any():
                mode_val = df_filled['motion_detected'].mode()[0] if not df_filled['motion_detected'].mode().empty else False
                df_filled['motion_detected'] = df_filled['motion_detected'].fillna(mode_val)
        
        return df_filled
    
    # Apply validation rules and flag invalid data
    def validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        df_validated = df.copy()
        
        # Count invalid entries
        invalid_count = 0
        
        for col, rules in self.validation_rules.items():
            if col in df_validated.columns:
                if rules['required'] and df_validated[col].isna().any():
                    missing_count = df_validated[col].isna().sum()
                    print(f"Warning: Required column {col} has {missing_count} missing values.")
                    invalid_count += missing_count
                
                if 'min' in rules:
                    below_min = df_validated[col] < rules['min']
                    if below_min.any():
                        count_below = below_min.sum()
                        print(f"  Found {count_below} values in {col} below minimum {rules['min']}")
                        df_validated.loc[below_min, col] = np.nan
                        invalid_count += count_below
                
                if 'max' in rules:
                    above_max = df_validated[col] > rules['max']
                    if above_max.any():
                        count_above = above_max.sum()
                        print(f"  Found {count_above} values in {col} above maximum {rules['max']}")
                        df_validated.loc[above_max, col] = np.nan
                        invalid_count += count_above

        # Log number of invalid records
        if invalid_count > 0:
            print(f"Total invalid entries after validation: {invalid_count}")

        return df_validated
    
    # DOMAIN-SPECIFIC outlier detection for child safety
    def handle_outliers_domain_specific(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df_clean = df.copy()
        outlier_report = {
            'temperature': {'anomalies': 0, 'details': []},
            'co2': {'anomalies': 0, 'details': []},
            'parked_duration': {'anomalies': 0, 'details': []}
        }
        
        # 1. Temperature anomalies based on safety thresholds
        if 'temperature_c' in df.columns:
            temp_series = df['temperature_c']
            
            # Method 1: Safety threshold outliers
            danger_temp = self.safety_thresholds['temperature_c']['danger']
            warning_temp = self.safety_thresholds['temperature_c']['warning']
            
            danger_outliers = temp_series > danger_temp
            warning_outliers = (temp_series > warning_temp) & (temp_series <= danger_temp)
            
            if danger_outliers.any():
                count = danger_outliers.sum()
                outlier_report['temperature']['anomalies'] += count
                outlier_report['temperature']['details'].append(
                    f"DANGER: {count} readings above {danger_temp}°C"
                )
                print(f"  Found {count} DANGEROUS temperature readings (> {danger_temp}°C)")
                
            if warning_outliers.any():
                count = warning_outliers.sum()
                outlier_report['temperature']['details'].append(
                    f"WARNING: {count} readings between {warning_temp}°C and {danger_temp}°C"
                )
                print(f"  Found {count} WARNING temperature readings ({warning_temp}°C - {danger_temp}°C)")
            
            # Method 2: Rate of change anomalies (sudden spikes/drops)
            if len(temp_series) > 5:
                temp_change = temp_series.diff().abs()
                change_threshold = temp_change.mean() + 2 * temp_change.std()
                
                rapid_changes = temp_change > change_threshold
                if rapid_changes.any():
                    count = rapid_changes.sum()
                    outlier_report['temperature']['details'].append(
                        f"RAPID CHANGE: {count} readings with change > {change_threshold:.1f}°C/min"
                    )
                    print(f"  Found {count} rapid temperature changes (> {change_threshold:.1f}°C/min)")
            
            # Method 3: Statistical outliers (IQR) for baseline comparison
            if temp_series.nunique() > 5:
                Q1 = temp_series.quantile(0.25)
                Q3 = temp_series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 3 * IQR  # Use 3*IQR for more conservative detection
                    upper_bound = Q3 + 3 * IQR
                    
                    statistical_outliers = (temp_series < lower_bound) | (temp_series > upper_bound)
                    if statistical_outliers.any():
                        count = statistical_outliers.sum()
                        outlier_report['temperature']['details'].append(
                            f"STATISTICAL: {count} readings outside [{lower_bound:.1f}, {upper_bound:.1f}]°C"
                        )
                        print(f"  Found {count} statistical temperature outliers")
        
        # 2. CO2 level anomalies
        if 'co2_level' in df.columns:
            co2_series = df['co2_level']
            danger_co2 = self.safety_thresholds['co2_level']['danger']
            warning_co2 = self.safety_thresholds['co2_level']['warning']
            
            danger_outliers = co2_series > danger_co2
            warning_outliers = (co2_series > warning_co2) & (co2_series <= danger_co2)
            
            if danger_outliers.any():
                count = danger_outliers.sum()
                outlier_report['co2']['anomalies'] = count
                outlier_report['co2']['details'].append(
                    f"DANGER: {count} readings above {danger_co2} ppm"
                )
                print(f"  Found {count} DANGEROUS CO2 readings (> {danger_co2} ppm)")
                
            if warning_outliers.any():
                count = warning_outliers.sum()
                outlier_report['co2']['details'].append(
                    f"WARNING: {count} readings between {warning_co2} ppm and {danger_co2} ppm"
                )
                print(f"  Found {count} WARNING CO2 readings ({warning_co2} ppm - {danger_co2} ppm)")
        
        # 3. Parked duration anomalies
        if 'engine_state' in df.columns and 'timestamp' in df.columns:
            # Calculate continuous parked time
            parked_segments = []
            current_segment = []
            
            for i, (state, time) in enumerate(zip(df['engine_state'], df['timestamp'])):
                if state == 'off':
                    current_segment.append(time)
                elif current_segment:
                    parked_segments.append(current_segment)
                    current_segment = []
            
            if current_segment:
                parked_segments.append(current_segment)
            
            # Check each parked segment duration
            max_allowed = self.safety_thresholds['max_parked_time']
            for segment in parked_segments:
                if segment:
                    duration = (segment[-1] - segment[0]).total_seconds() / 60  # minutes
                    if duration > max_allowed:
                        outlier_report['parked_duration']['anomalies'] += 1
                        outlier_report['parked_duration']['details'].append(
                            f"Parked for {duration:.1f} minutes (exceeds {max_allowed} min)"
                        )
                        print(f"  Found parked duration anomaly: {duration:.1f} minutes")
        
        return df_clean, outlier_report
    
    # Create new features for analysis
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_features = df.copy()
        
        # Time-based features
        if 'timestamp' in df.columns:
            df_features['hour'] = df['timestamp'].dt.hour
            df_features['minute'] = df['timestamp'].dt.minute
            df_features['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Temperature features
        if 'temperature_c' in df.columns:
            # Rate of change
            df_features['temp_change'] = df['temperature_c'].diff().fillna(0)
            df_features['temp_acceleration'] = df_features['temp_change'].diff().fillna(0)
            
            # Safety flags
            danger_temp = self.safety_thresholds['temperature_c']['danger']
            warning_temp = self.safety_thresholds['temperature_c']['warning']
            
            df_features['temp_danger_flag'] = (df['temperature_c'] > danger_temp).astype(int)
            df_features['temp_warning_flag'] = (
                (df['temperature_c'] > warning_temp) & 
                (df['temperature_c'] <= danger_temp)
            ).astype(int)
            
            # Risk scores (0-1)
            df_features['temp_risk'] = np.clip((df['temperature_c'] - warning_temp) / (danger_temp - warning_temp), 0, 1)
        
        # Occupancy detection
        if all(col in df.columns for col in ['weight_left_kg', 'weight_right_kg']):
            df_features['total_weight'] = df_features['weight_left_kg'] + df_features['weight_right_kg']
            df_features['occupancy'] = (df_features['total_weight'] > 5).astype(int)
            
            # Child-specific occupancy (lighter weight on one seat)
            df_features['child_detected'] = (
                (df['weight_left_kg'] > 2) & (df['weight_left_kg'] < 25) & 
                (df['weight_right_kg'] < 5)
            ).astype(int)
            
            # Adult detection
            df_features['adult_present'] = (df['weight_right_kg'] > 50).astype(int)
        
        # CO2 safety features
        if 'co2_level' in df.columns:
            danger_co2 = self.safety_thresholds['co2_level']['danger']
            warning_co2 = self.safety_thresholds['co2_level']['warning']
            
            df_features['co2_danger_flag'] = (df['co2_level'] > danger_co2).astype(int)
            df_features['co2_warning_flag'] = (
                (df['co2_level'] > warning_co2) & 
                (df['co2_level'] <= danger_co2)
            ).astype(int)
            
            df_features['co2_risk'] = np.clip((df['co2_level'] - warning_co2) / (danger_co2 - warning_co2), 0, 1)
        
        # Combined risk score
        risk_cols = [col for col in ['temp_risk', 'co2_risk'] if col in df_features.columns]
        if risk_cols:
            df_features['overall_risk'] = df_features[risk_cols].max(axis=1)  # Use max for safety
            
            # Emergency flag (if any critical danger)
            danger_flags = [col for col in ['temp_danger_flag', 'co2_danger_flag'] if col in df_features.columns]
            if danger_flags:
                df_features['emergency_flag'] = df_features[danger_flags].max(axis=1)
        
        # Time since engine off (if child might be left behind)
        if 'engine_state' in df.columns and 'timestamp' in df.columns:
            engine_off_time = None
            parked_duration = []
            
            for i, (state, time) in enumerate(zip(df['engine_state'], df['timestamp'])):
                if state == 'off':
                    if engine_off_time is None:
                        engine_off_time = time
                    duration = (time - engine_off_time).total_seconds() / 60
                else:
                    engine_off_time = None
                    duration = 0
                
                parked_duration.append(duration)
            
            df_features['parked_duration_min'] = parked_duration
            
            # Parked too long flag
            max_allowed = self.safety_thresholds['max_parked_time']
            df_features['parked_too_long'] = (df_features['parked_duration_min'] > max_allowed).astype(int)
        
        return df_features
    
    # Generate a data quality report
    def save_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                            outlier_report: Dict, file_path: str):
        
        report = {
            'original_records': len(original_df),
            'cleaned_records': len(cleaned_df),
            'records_removed': len(original_df) - len(cleaned_df),
            'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df) * 100),
            'missing_values_before': original_df.isnull().sum().to_dict(),
            'missing_values_after': cleaned_df.isnull().sum().to_dict(),
            'data_types': {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
            'summary_statistics': self.get_safe_statistics(cleaned_df),
            'outlier_analysis': outlier_report,
            'safety_analysis': self.get_safety_analysis(cleaned_df)
        }
        
        # Save report to processed folder
        os.makedirs(os.path.dirname(file_path.replace('raw', 'processed')), exist_ok=True)
        report_path = file_path.replace('raw', 'processed').replace('.csv', '_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  Saved cleaning report to {report_path}")
        
        # Also save the cleaned data
        cleaned_file = file_path.replace('raw', 'processed').replace('.csv', '_cleaned.csv')
        cleaned_df.to_csv(cleaned_file, index=False)
        print(f"  Saved cleaned data to {cleaned_file}")

    # Safely compute statistics avoiding NaN std issues
    def get_safe_statistics(self, df: pd.DataFrame) -> Dict:
            
            stats_dict = {}
            
            # Only calculate statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = df[col].dropna()  # Drop NaN values
                
                if len(col_data) == 0:
                    # No valid data
                    stats_dict[col] = {
                        'count': 0,
                        'mean': None,
                        'std': None,
                        'min': None,
                        '25%': None,
                        '50%': None,
                        '75%': None,
                        'max': None
                    }
                elif len(col_data) == 1:
                    # Only one value, std is 0 not NaN
                    val = float(col_data.iloc[0])
                    stats_dict[col] = {
                        'count': len(col_data),
                        'mean': val,
                        'std': 0.0,  # Standard deviation is 0 for single value
                        'min': val,
                        '25%': val,
                        '50%': val,
                        '75%': val,
                        'max': val
                    }
                else:
                    # Multiple values, calculate normally
                    try:
                        desc = col_data.describe().to_dict()
                        # Ensure std is a number, not NaN
                        if pd.isna(desc.get('std')):
                            desc['std'] = 0.0
                        stats_dict[col] = desc
                    except:
                        # Fallback if describe fails
                        stats_dict[col] = {
                            'count': len(col_data),
                            'mean': float(col_data.mean()) if len(col_data) > 0 else None,
                            'std': 0.0 if len(col_data) <= 1 else float(col_data.std()),
                            'min': float(col_data.min()) if len(col_data) > 0 else None,
                            'max': float(col_data.max()) if len(col_data) > 0 else None
                        }
            
            return stats_dict
    
    def get_safety_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze safety-related aspects of the data"""
        analysis = {
            'temperature_safety': {},
            'co2_safety': {},
            'occupancy_status': {},
            'risk_assessment': {}
        }
        
        # Temperature safety analysis
        if 'temperature_c' in df.columns:
            danger_temp = self.safety_thresholds['temperature_c']['danger']
            warning_temp = self.safety_thresholds['temperature_c']['warning']
            
            analysis['temperature_safety'] = {
                'danger_count': int((df['temperature_c'] > danger_temp).sum()),
                'warning_count': int(((df['temperature_c'] > warning_temp) & 
                                      (df['temperature_c'] <= danger_temp)).sum()),
                'max_temperature': float(df['temperature_c'].max()),
                'min_temperature': float(df['temperature_c'].min()),
                'avg_temperature': float(df['temperature_c'].mean())
            }
        
        # CO2 safety analysis
        if 'co2_level' in df.columns:
            danger_co2 = self.safety_thresholds['co2_level']['danger']
            warning_co2 = self.safety_thresholds['co2_level']['warning']
            
            analysis['co2_safety'] = {
                'danger_count': int((df['co2_level'] > danger_co2).sum()),
                'warning_count': int(((df['co2_level'] > warning_co2) & 
                                     (df['co2_level'] <= danger_co2)).sum()),
                'max_co2': float(df['co2_level'].max()),
                'avg_co2': float(df['co2_level'].mean())
            }
        
        # Occupancy analysis
        if 'total_weight' in df.columns:
            analysis['occupancy_status'] = {
                'occupied_count': int((df['total_weight'] > 5).sum()),
                'max_weight': float(df['total_weight'].max()),
                'child_detected_count': int(df['child_detected'].sum()) if 'child_detected' in df.columns else 0
            }
        
        # Risk assessment
        risk_scores = []
        if 'overall_risk' in df.columns:
            risk_scores = df['overall_risk'].tolist()
        
        analysis['risk_assessment'] = {
            'high_risk_count': int((df['overall_risk'] > 0.7).sum()) if 'overall_risk' in df.columns else 0,
            'medium_risk_count': int(((df['overall_risk'] > 0.3) & (df['overall_risk'] <= 0.7)).sum()) 
                               if 'overall_risk' in df.columns else 0,
            'low_risk_count': int((df['overall_risk'] <= 0.3).sum()) if 'overall_risk' in df.columns else 0,
            'max_risk': float(df['overall_risk'].max()) if 'overall_risk' in df.columns else 0,
            'avg_risk': float(df['overall_risk'].mean()) if 'overall_risk' in df.columns else 0
        }
        
        return analysis
    
    # Create Scatterplot for visualizing outliers
    def plot_outliers(self, df: pd.DataFrame, column: str, save_path: str = None):
        if column not in df.columns:
            print(f"Column {column} not found in DataFrame.")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Create time-based index for better visualization
        if 'timestamp' in df.columns:
            x_values = df['timestamp']
            x_label = 'Time'
        else:
            x_values = df.index
            x_label = 'Index'
        
        # Plot scatter
        plt.scatter(x_values, df[column], alpha=0.7, label='Data points', color='blue', s=50)
        
        # Plot safety thresholds
        if column == 'temperature_c':
            danger_temp = self.safety_thresholds['temperature_c']['danger']
            warning_temp = self.safety_thresholds['temperature_c']['warning']
            
            plt.axhline(y=danger_temp, color='red', linestyle='-', linewidth=2, 
                       alpha=0.8, label='Danger Threshold')
            plt.axhline(y=warning_temp, color='orange', linestyle='--', linewidth=2, 
                       alpha=0.6, label='Warning Threshold')
            
            # Highlight dangerous points
            danger_points = df[df[column] > danger_temp]
            if not danger_points.empty:
                plt.scatter(danger_points['timestamp'] if 'timestamp' in df.columns else danger_points.index,
                           danger_points[column], color='red', s=100, 
                           alpha=1.0, label='Dangerous', edgecolors='black', linewidth=2)
            
            # Highlight warning points
            warning_points = df[(df[column] > warning_temp) & (df[column] <= danger_temp)]
            if not warning_points.empty:
                plt.scatter(warning_points['timestamp'] if 'timestamp' in df.columns else warning_points.index,
                           warning_points[column], color='orange', s=80, 
                           alpha=0.8, label='Warning', edgecolors='black', linewidth=1)
        
        elif column == 'co2_level':
            danger_co2 = self.safety_thresholds['co2_level']['danger']
            warning_co2 = self.safety_thresholds['co2_level']['warning']
            
            plt.axhline(y=danger_co2, color='red', linestyle='-', linewidth=2, 
                       alpha=0.8, label='Danger Threshold')
            plt.axhline(y=warning_co2, color='orange', linestyle='--', linewidth=2, 
                       alpha=0.6, label='Warning Threshold')
        
        plt.title(f'{column} - Safety Monitoring', fontsize=16, fontweight='bold')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(column, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"  Safety plot saved to {save_path}")
        
        plt.show()


# Main script to run the pipeline
if __name__ == "__main__":
    # Create and clean data
    from sensor_simulator import SensorSimulator
    
    print("Data Cleaning Pipeline")
    print("--------Started---------")

    # Step 1: Generate data
    simulator = SensorSimulator()
    raw_df = simulator.simulate_one_hour()
    
    # Step 2: Clean data
    cleaner = DataCleaningPipeline()
    cleaned_df = cleaner.clean_sensor_data("data/raw/sensor_data_hour.csv")
    
    # Step 3: Save cleaned data
    os.makedirs("data/processed", exist_ok=True)
    cleaned_df.to_csv("data/processed/cleaned_sensor_data.csv", index=False)
    print(f"\nCleaned data saved to: data/processed/cleaned_sensor_data.csv")

    # Display safety plot for temperature
    plot_path = "data/processed/temperature_safety_plot.png"
    cleaner.plot_outliers(cleaned_df, 'temperature_c', plot_path)
    
    # Show sample of cleaned data with safety features
    print("\nSample of cleaned data with safety features:")
    safety_cols = [col for col in cleaned_df.columns if 'risk' in col or 'flag' in col or 'danger' in col or 'warning' in col]
    display_cols = ['timestamp', 'temperature_c', 'co2_level'] + safety_cols[:5]
    display_cols = [col for col in display_cols if col in cleaned_df.columns]
    print(cleaned_df[display_cols].head())
    
    # Print safety summary
    print("\nSafety Summary:")
    if 'temp_danger_flag' in cleaned_df.columns:
        print(f"  Dangerous temperature readings: {cleaned_df['temp_danger_flag'].sum()}")
    if 'temp_warning_flag' in cleaned_df.columns:
        print(f"  Warning temperature readings: {cleaned_df['temp_warning_flag'].sum()}")
    if 'emergency_flag' in cleaned_df.columns:
        print(f"  Emergency situations detected: {cleaned_df['emergency_flag'].sum()}")