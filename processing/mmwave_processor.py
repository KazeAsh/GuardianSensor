import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import os

class MMWaveProcessor:
    """
    Real-time mmWave radar signal processing pipeline
    Implements FMCW radar processing techniques for vital sign detection
    """
    
    def __init__(self, sampling_rate=100, fft_size=1024):
        self.fs = sampling_rate  # Sampling frequency
        self.fft_size = fft_size
        
        # Vital sign frequency ranges (Hz)
        self.breathing_range = (0.1, 0.5)    # 6-30 breaths/min
        self.heartbeat_range = (0.8, 3.0)    # 48-180 BPM
        
        # Initialize filters
        self._init_filters()
        
        # Detection thresholds
        self.breathing_threshold = 0.1
        self.heartbeat_threshold = 0.05
        
    def _init_filters(self):
        """Initialize digital filters for signal processing"""
        # Bandpass for breathing
        self.breathing_sos = signal.butter(
            4, self.breathing_range, btype='band', 
            fs=self.fs, output='sos'
        )
        
        # Bandpass for heartbeat
        self.heartbeat_sos = signal.butter(
            4, self.heartbeat_range, btype='band', 
            fs=self.fs, output='sos'
        )
        
        # Notch filter for powerline interference (50/60 Hz)
        b, a = signal.iirnotch(50, 30, self.fs)  # Remove 50 Hz interference
        self.notch_sos = signal.tf2sos(b, a)
    
    def process_iq_data(self, iq_data: np.ndarray) -> Dict:
        """
        Complete processing pipeline for mmWave I/Q data
        Returns vital signs and detection confidence
        """
        # 1. Preprocessing
        cleaned_iq = self._preprocess_iq(iq_data)
        
        # 2. Extract phase information (vital signs are in phase variations)
        phase_signal = self._extract_phase(cleaned_iq)
        
        # 3. Apply vital sign filters
        breathing_signal = signal.sosfilt(self.breathing_sos, phase_signal)
        heartbeat_signal = signal.sosfilt(self.heartbeat_sos, phase_signal)
        
        # 4. Remove DC offset and trends
        breathing_signal = signal.detrend(breathing_signal)
        heartbeat_signal = signal.detrend(heartbeat_signal)
        
        # 5. Frequency domain analysis
        breathing_fft, breathing_freqs = self._compute_spectrum(breathing_signal)
        heartbeat_fft, heartbeat_freqs = self._compute_spectrum(heartbeat_signal)
        
        # 6. Peak detection in frequency domain
        vital_signs = self._detect_vital_signs(
            breathing_fft, breathing_freqs,
            heartbeat_fft, heartbeat_freqs
        )
        
        # 7. Quality assessment
        quality_metrics = self._assess_signal_quality(
            breathing_signal, heartbeat_signal,
            breathing_fft, heartbeat_fft
        )
        
        # 8. Motion artifact detection
        motion_artifact = self._detect_motion_artifacts(iq_data)
        
        return {
            'vital_signs': vital_signs,
            'quality_metrics': quality_metrics,
            'motion_artifact': motion_artifact,
            'breathing_signal': breathing_signal.tolist()[-100:],  # Last 100 samples
            'heartbeat_signal': heartbeat_signal.tolist()[-100:],
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _preprocess_iq(self, iq_data: np.ndarray) -> np.ndarray:
        """Preprocess I/Q data: filtering, normalization"""
        # Convert to numpy array if needed
        iq_array = np.array(iq_data, dtype=complex)
        
        # Remove DC offset
        iq_centered = iq_array - np.mean(iq_array)
        
        # Apply notch filter to remove interference
        iq_filtered = signal.sosfilt(self.notch_sos, iq_centered)
        
        # Normalize amplitude
        iq_normalized = iq_filtered / np.max(np.abs(iq_filtered)) if np.max(np.abs(iq_filtered)) > 0 else iq_filtered
        
        return iq_normalized
    
    def _extract_phase(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Extract phase information from I/Q data
        Vital signs cause tiny phase variations in radar signal
        """
        # Phase = arctan(Q/I)
        phase = np.unwrap(np.angle(iq_data))
        
        # Remove linear trend (range information)
        phase_detrended = signal.detrend(phase)
        
        return phase_detrended
    
    def _compute_spectrum(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum using FFT"""
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(signal_data))
        windowed_signal = signal_data * window
        
        # Compute FFT
        n = len(windowed_signal)
        fft_result = fft(windowed_signal, n=self.fft_size)
        
        # Compute power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_result[:self.fft_size//2]) ** 2
        
        # Frequency bins
        freqs = fftfreq(self.fft_size, 1/self.fs)[:self.fft_size//2]
        
        return power_spectrum, freqs
    
    def _detect_vital_signs(self, breathing_spectrum, breathing_freqs,
                           heartbeat_spectrum, heartbeat_freqs) -> Dict:
        """Detect breathing and heartbeat from frequency spectra"""
        
        # Find breathing rate
        breathing_idx = np.where((breathing_freqs >= self.breathing_range[0]) & 
                                (breathing_freqs <= self.breathing_range[1]))[0]
        
        if len(breathing_idx) > 0:
            breathing_power = breathing_spectrum[breathing_idx]
            breathing_peak_idx = np.argmax(breathing_power)
            breathing_freq = breathing_freqs[breathing_idx[breathing_peak_idx]]
            breathing_bpm = breathing_freq * 60
            breathing_confidence = min(breathing_power[breathing_peak_idx] / np.max(breathing_spectrum), 1.0)
        else:
            breathing_bpm = 0
            breathing_confidence = 0
        
        # Find heartbeat rate
        heartbeat_idx = np.where((heartbeat_freqs >= self.heartbeat_range[0]) & 
                                (heartbeat_freqs <= self.heartbeat_range[1]))[0]
        
        if len(heartbeat_idx) > 0:
            heartbeat_power = heartbeat_spectrum[heartbeat_idx]
            heartbeat_peak_idx = np.argmax(heartbeat_power)
            heartbeat_freq = heartbeat_freqs[heartbeat_idx[heartbeat_peak_idx]]
            heartbeat_bpm = heartbeat_freq * 60
            heartbeat_confidence = min(heartbeat_power[heartbeat_peak_idx] / np.max(heartbeat_spectrum), 1.0)
        else:
            heartbeat_bpm = 0
            heartbeat_confidence = 0
        
        # Determine if vital signs are detected
        vital_signs_detected = (
            breathing_confidence > self.breathing_threshold or 
            heartbeat_confidence > self.heartbeat_threshold
        )
        
        # Classify as child or adult based on rates
        if heartbeat_bpm > 0:
            if heartbeat_bpm > 100:  # Typical child heart rate
                occupant_type = 'child'
                type_confidence = min((heartbeat_bpm - 100) / 20, 1.0)
            else:
                occupant_type = 'adult'
                type_confidence = min((100 - heartbeat_bpm) / 40, 1.0)
        else:
            occupant_type = 'unknown'
            type_confidence = 0
        
        return {
            'breathing_rate_bpm': round(breathing_bpm, 1),
            'heart_rate_bpm': round(heartbeat_bpm, 1),
            'breathing_confidence': round(breathing_confidence, 2),
            'heartbeat_confidence': round(heartbeat_confidence, 2),
            'vital_signs_detected': vital_signs_detected,
            'occupant_type': occupant_type,
            'type_confidence': round(type_confidence, 2),
            'breathing_frequency_hz': round(breathing_freq, 3) if 'breathing_freq' in locals() else 0,
            'heartbeat_frequency_hz': round(heartbeat_freq, 3) if 'heartbeat_freq' in locals() else 0
        }
    
    def _assess_signal_quality(self, breathing_signal, heartbeat_signal,
                              breathing_spectrum, heartbeat_spectrum) -> Dict:
        """Assess quality of detected signals"""
        
        # Signal-to-noise ratio (simplified)
        breathing_snr = self._estimate_snr(breathing_signal)
        heartbeat_snr = self._estimate_snr(heartbeat_signal)
        
        # Spectral purity (peak prominence)
        breathing_purity = self._spectral_purity(breathing_spectrum)
        heartbeat_purity = self._spectral_purity(heartbeat_spectrum)
        
        return {
            'breathing_snr_db': round(breathing_snr, 1),
            'heartbeat_snr_db': round(heartbeat_snr, 1),
            'breathing_spectral_purity': round(breathing_purity, 2),
            'heartbeat_spectral_purity': round(heartbeat_purity, 2),
            'overall_quality': round((breathing_snr/20 + heartbeat_snr/20 + 
                                    breathing_purity + heartbeat_purity) / 4, 2)
        }
    
    def _estimate_snr(self, signal_data):
        """Estimate signal-to-noise ratio"""
        if len(signal_data) == 0:
            return 0
        
        # Simple SNR estimation
        signal_power = np.mean(signal_data ** 2)
        noise_power = np.var(signal_data - signal.medfilt(signal_data, 5))
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return max(snr, 0)
        return 0
    
    def _spectral_purity(self, spectrum):
        """Measure how concentrated the spectrum is around peaks"""
        if len(spectrum) == 0 or np.sum(spectrum) == 0:
            return 0
        
        normalized_spectrum = spectrum / np.sum(spectrum)
        # Entropy measure (lower entropy = purer spectrum)
        entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
        max_entropy = np.log2(len(spectrum))
        
        purity = 1 - (entropy / max_entropy)
        return max(0, min(1, purity))
    
    def _detect_motion_artifacts(self, iq_data):
        """Detect large movements that could interfere with vital signs"""
        amplitude = np.abs(iq_data)
        
        # Calculate movement index (variation in amplitude)
        amplitude_diff = np.diff(amplitude)
        movement_index = np.mean(np.abs(amplitude_diff))
        
        # Detect sudden movements
        movement_threshold = 0.1
        has_motion_artifact = movement_index > movement_threshold
        
        return {
            'movement_index': round(movement_index, 3),
            'has_motion_artifact': has_motion_artifact,
            'movement_severity': 'high' if movement_index > 0.2 else 
                                'medium' if movement_index > 0.1 else 'low'
        }
    
    def batch_process_dataset(self, dataset_path):
        """Process entire dataset and generate analysis"""
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        results = []
        
        print(f"Processing {len(dataset['scenarios'])} scenarios...")
        
        for scenario in dataset['scenarios']:
            # Load I/Q data if available
            iq_file = f"data/raw/mmwave/{scenario['scenario_id']}_iq.json"
            if os.path.exists(iq_file):
                with open(iq_file, 'r') as f:
                    iq_data = json.load(f)
                
                # Reconstruct complex I/Q data
                iq_real = np.array(iq_data['iq_real'])
                iq_imag = np.array(iq_data['iq_imag'])
                iq_complex = iq_real + 1j * iq_imag
                
                # Process the data
                processing_result = self.process_iq_data(iq_complex)
                
                # Combine with scenario data
                result = {
                    'scenario_id': scenario['scenario_id'],
                    'ground_truth': {
                        'has_child': scenario['has_child'],
                        'movement_level': scenario['movement_level']
                    },
                    'processing_result': processing_result,
                    'car_sensors': scenario['car_sensors'],
                    'timestamp': scenario['timestamp']
                }
                
                results.append(result)
        
        # Save processing results
        output_path = dataset_path.replace('.json', '_processed.json')
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': dataset['metadata'],
                'processing_metadata': {
                    'processor_version': '1.0',
                    'processing_date': pd.Timestamp.now().isoformat(),
                    'sampling_rate': self.fs,
                    'fft_size': self.fft_size
                },
                'results': results
            }, f, indent=2, default=str)
        
        print(f"Processing complete. Results saved to {output_path}")
        return results
    
    def visualize_processing(self, iq_data, save_path=None):
        """Create visualization of signal processing pipeline"""
        # Process data
        result = self.process_iq_data(iq_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # Time: Raw I/Q amplitude
        time_axis = np.arange(len(iq_data)) / self.fs
        axes[0, 0].plot(time_axis, np.abs(iq_data))
        axes[0, 0].set_title('Raw Radar Signal (Amplitude)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # Time: Extracted phase
        phase = self._extract_phase(iq_data)
        axes[0, 1].plot(time_axis[:len(phase)], phase)
        axes[0, 1].set_title('Extracted Phase (Vital Signs)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Phase (rad)')
        axes[0, 1].grid(True)
        
        # Time: Filtered breathing signal
        breathing_signal = signal.sosfilt(self.breathing_sos, phase)
        axes[1, 0].plot(time_axis[:len(breathing_signal)], breathing_signal)
        axes[1, 0].set_title('Filtered Breathing Signal')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True)
        
        # Time: Filtered heartbeat signal
        heartbeat_signal = signal.sosfilt(self.heartbeat_sos, phase)
        axes[1, 1].plot(time_axis[:len(heartbeat_signal)], heartbeat_signal)
        axes[1, 1].set_title('Filtered Heartbeat Signal')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True)
        
        # Frequency: Breathing spectrum
        breathing_fft, breathing_freqs = self._compute_spectrum(breathing_signal)
        axes[2, 0].plot(breathing_freqs, breathing_fft)
        axes[2, 0].set_title('Breathing Spectrum')
        axes[2, 0].set_xlabel('Frequency (Hz)')
        axes[2, 0].set_ylabel('Power')
        axes[2, 0].grid(True)
        
        # Frequency: Heartbeat spectrum
        heartbeat_fft, heartbeat_freqs = self._compute_spectrum(heartbeat_signal)
        axes[2, 1].plot(heartbeat_freqs, heartbeat_fft)
        axes[2, 1].set_title('Heartbeat Spectrum')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('Power')
        axes[2, 1].grid(True)
        
        # Add vital signs text
        vital_text = (
            f"Breathing: {result['vital_signs']['breathing_rate_bpm']} BPM "
            f"(conf: {result['vital_signs']['breathing_confidence']:.2f})\n"
            f"Heartbeat: {result['vital_signs']['heart_rate_bpm']} BPM "
            f"(conf: {result['vital_signs']['heartbeat_confidence']:.2f})\n"
            f"Occupant: {result['vital_signs']['occupant_type']} "
            f"(conf: {result['vital_signs']['type_confidence']:.2f})"
        )
        plt.figtext(0.5, 0.01, vital_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        return fig

# Main processing script
if __name__ == "__main__":
    print("📡 GuardianSensor: mmWave Signal Processing Pipeline")
    print("=" * 60)
    
    # Initialize processor
    processor = MMWaveProcessor(sampling_rate=100, fft_size=1024)
    
    # Generate test data
    from utils.mmwave_simulator import MMWaveSimulator
    simulator = MMWaveSimulator(sampling_rate=100, duration=30)
    iq_data = simulator.generate_mmwave_iq_data(has_child=True, movement_level='low')
    
    # Process the data
    print("\n1. Processing mmWave radar data...")
    result = processor.process_iq_data(iq_data)
    
    print(f"\n2. Vital Signs Detection Results:")
    vital = result['vital_signs']
    print(f"   Breathing Rate: {vital['breathing_rate_bpm']} BPM (confidence: {vital['breathing_confidence']})")
    print(f"   Heart Rate: {vital['heart_rate_bpm']} BPM (confidence: {vital['heartbeat_confidence']})")
    print(f"   Occupant Type: {vital['occupant_type']} (confidence: {vital['type_confidence']})")
    print(f"   Vital Signs Detected: {vital['vital_signs_detected']}")
    
    print(f"\n3. Signal Quality:")
    quality = result['quality_metrics']
    print(f"   Breathing SNR: {quality['breathing_snr_db']} dB")
    print(f"   Heartbeat SNR: {quality['heartbeat_snr_db']} dB")
    print(f"   Overall Quality: {quality['overall_quality']}")
    
    print(f"\n4. Motion Analysis:")
    motion = result['motion_artifact']
    print(f"   Movement Index: {motion['movement_index']}")
    print(f"   Has Motion Artifact: {motion['has_motion_artifact']}")
    print(f"   Severity: {motion['movement_severity']}")
    
    # Create visualization
    print("\n5. Creating signal processing visualization...")
    os.makedirs('outputs/visualizations', exist_ok=True)
    processor.visualize_processing(iq_data, 'outputs/visualizations/mmwave_processing.png')
    
    # Process entire dataset if available
    dataset_path = 'data/processed/mmwave_vital_signs.json'
    if os.path.exists(dataset_path):
        print(f"\n6. Batch processing dataset...")
        results = processor.batch_process_dataset(dataset_path)
        print(f"   Processed {len(results)} scenarios")
    
    print("\n✅ Signal processing pipeline complete!")