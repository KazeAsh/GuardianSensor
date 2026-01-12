# processing/compatible_processor.py
"""
Compatible MMWave signal processor for vital sign detection.

This module provides signal processing capabilities for millimeter-wave
radar data, compatible with SciPy 1.10.1.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


class CompatibleMMWaveProcessor:
    """Signal processor for MMWave vital sign detection.
    
    Processes I/Q data from millimeter-wave radar to extract breathing
    and heart rate measurements using bandpass and notch filtering.
    """
    
    # Signal processing constants
    BREATHING_FREQ_RANGE = (0.1, 0.5)  # Hz
    HEARTBEAT_FREQ_RANGE = (0.8, 3.0)  # Hz
    NOTCH_FREQ = 50  # Hz (power line interference)
    NOTCH_QUALITY = 30
    
    def __init__(self, sampling_rate: int = 100):
        """Initialize processor with specified sampling rate.
        
        Args:
            sampling_rate: Sampling frequency in Hz. Defaults to 100.
        """
        self.fs = sampling_rate
        self._init_filters()
    
    def _init_filters(self) -> None:
        """Initialize digital filters for signal processing."""
        try:
            # Breathing detection filter
            self.breathing_b, self.breathing_a = signal.butter(
                4, self.BREATHING_FREQ_RANGE, btype='band', fs=self.fs
            )
            
            # Heart rate detection filter
            self.heartbeat_b, self.heartbeat_a = signal.butter(
                4, self.HEARTBEAT_FREQ_RANGE, btype='band', fs=self.fs
            )
            
            # Notch filter for power line interference
            self.notch_b, self.notch_a = signal.iirnotch(
                self.NOTCH_FREQ, self.NOTCH_QUALITY, self.fs
            )
        except Exception as e:
            logger.error(f"Filter initialization failed: {e}")
            raise
    
    def process_iq_data(self, iq_data: np.ndarray) -> Dict[str, Any]:
        """Process I/Q radar data to extract vital signs.
        
        Args:
            iq_data: Complex I/Q signal array.
            
        Returns:
            Dictionary containing vital signs and processing metadata.
            
        Raises:
            ValueError: If input data is invalid.
        """
        if not isinstance(iq_data, (np.ndarray, list)) or len(iq_data) == 0:
            raise ValueError("Invalid I/Q data input")
        
        try:
            # Preprocessing
            iq_array = np.array(iq_data, dtype=complex)
            iq_centered = iq_array - np.mean(iq_array)
            
            # Apply notch filter
            iq_filtered = signal.filtfilt(self.notch_b, self.notch_a, iq_centered)
            
            # Extract amplitude
            amplitude = np.abs(iq_filtered)
            
            # Apply vital sign filters
            breathing = signal.filtfilt(self.breathing_b, self.breathing_a, amplitude)
            heartbeat = signal.filtfilt(self.heartbeat_b, self.heartbeat_a, amplitude)
            
            # Extract vital sign metrics
            breathing_bpm, breathing_conf = self._extract_vital_sign(breathing, 0.8)
            heartbeat_bpm, heartbeat_conf = self._extract_vital_sign(heartbeat, 0.3)
            
            return {
                'vital_signs': {
                    'breathing_rate_bpm': round(breathing_bpm, 1),
                    'heart_rate_bpm': round(heartbeat_bpm, 1),
                    'breathing_confidence': breathing_conf,
                    'heartbeat_confidence': heartbeat_conf,
                    'vital_signs_detected': breathing_bpm > 5 or heartbeat_bpm > 40
                },
                'status': 'success',
                'samples_processed': len(iq_data)
            }
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extract_vital_sign(self, signal_data: np.ndarray, 
                            min_distance_factor: float) -> Tuple[float, float]:
        """Extract vital sign rate and confidence.
        
        Args:
            signal_data: Filtered signal.
            min_distance_factor: Peak detection distance factor.
            
        Returns:
            Tuple of (rate_bpm, confidence_score).
        """
        peaks, _ = signal.find_peaks(signal_data, distance=self.fs * min_distance_factor)
        
        if len(peaks) < 2:
            return 0.0, 0.0
        
        duration_minutes = len(signal_data) / self.fs / 60
        bpm = len(peaks) / duration_minutes
        confidence = min(len(peaks) * 0.1, 1.0)
        
        return bpm, confidence