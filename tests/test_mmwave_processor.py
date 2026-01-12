import pytest
import numpy as np
from processing.mmwave_processor import MMWaveProcessor


class TestMMWaveProcessor:
    """Unit tests for MMWaveProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Fixture providing a configured MMWaveProcessor instance."""
        return MMWaveProcessor(sampling_rate=100, fft_size=1024)
    
    def test_initialization(self):
        """Test that processor initializes with correct parameters."""
        processor = MMWaveProcessor(sampling_rate=100, fft_size=1024)
        assert processor.fs == 100
        assert processor.fft_size == 1024
    
    def test_preprocess_iq(self, processor):
        """Test I/Q data preprocessing normalizes input correctly."""
        iq_data = np.random.randn(1000) + 1j * np.random.randn(1000)
        result = processor._preprocess_iq(iq_data)
        
        assert result.shape == iq_data.shape
        assert np.abs(np.mean(result)) < 0.1
    
    def test_vital_sign_detection(self):
        """Test vital sign detection produces expected output structure."""
        from utils.mmwave_simulator import MMWaveSimulator
        
        simulator = MMWaveSimulator(sampling_rate=100, duration=30)
        iq_data = simulator.generate_mmwave_iq_data(has_child=True)
        
        processor = MMWaveProcessor(sampling_rate=100)
        result = processor.process_iq_data(iq_data)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['vital_signs', 'quality_metrics', 'motion_artifact'])