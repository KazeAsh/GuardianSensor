#!/usr/bin/env python3
"""
GuardianSensor - Main Entry Point
mmWave Radar Child Safety System

Usage:
    python run.py [command]

Commands:
    api           Start the FastAPI API server
    dashboard     Start the Streamlit dashboard
    process       Run signal processing pipeline
    simulate      Generate simulation data
    test          Run test suite
    help          Show this help message
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║               🚗 GuardianSensor v2.0                     ║
    ║                                                          ║
    ║    mmWave Radar Child Safety System                      ║
    ║    Built for a better future for all                     ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    A privacy-first child safety system using mmWave radar technology
    for vital sign detection in vehicles.
    
    Repository: https://github.com/KazeAsh/GuardianSensor
    """
    print(banner)

def start_api_server(host="0.0.0.0", port=8000, reload=True):
    """Start FastAPI server"""
    print(f"🚀 Starting GuardianSensor API server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print(f"   Docs: http://{host}:{port}/docs")
    print(f"   Health: http://{host}:{port}/health")
    print()
    
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

def start_dashboard(port=8501):
    """Start Streamlit dashboard"""
    print(f"📊 Starting GuardianSensor Dashboard...")
    print(f"   Port: {port}")
    print(f"   URL: http://localhost:{port}")
    print()
    
    dashboard_path = project_root / "dashboard" / "app.py"
    subprocess.run([
        "streamlit", "run", str(dashboard_path),
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ])

def run_signal_processing():
    """Run signal processing pipeline - FIXED VERSION"""
    print("🔬 Running mmWave signal processing pipeline...")
    
    try:
        # Try compatible processor first
        try:
            from processing.compatible_processor import CompatibleMMWaveProcessor
            from utils.mmwave_simulator import MMWaveSimulator
            
            print("   Using compatible processor...")
            
            # Generate test data
            simulator = MMWaveSimulator(sampling_rate=100, duration=30)
            iq_data = simulator.generate_mmwave_iq_data(has_child=True, movement_level='low')
            
            # Process data
            processor = CompatibleMMWaveProcessor(sampling_rate=100)
            result = processor.process_iq_data(iq_data)
            
        except ImportError:
            # Fallback to basic simulator
            from utils.mmwave_simulator import MMWaveSimulator
            import numpy as np
            
            print("   Using basic simulator...")
            simulator = MMWaveSimulator(sampling_rate=100, duration=30)
            iq_data = simulator.generate_mmwave_iq_data(has_child=True)
            vital_signs = simulator.extract_vital_signs(iq_data)
            result = {'vital_signs': vital_signs, 'status': 'simulated'}
        
        # Show results
        print(f"   ✅ Processing complete!")
        vital = result.get('vital_signs', {})
        if isinstance(vital, dict):
            print(f"   Results: {vital}")
        else:
            print(f"   Results available")
        
        # Simple visualization
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            
            os.makedirs("outputs/visualizations", exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            plt.plot(np.abs(iq_data)[:500])
            plt.title('mmWave Radar Signal (First 500 samples)')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.savefig('outputs/visualizations/mmwave_simple.png', dpi=150)
            print(f"   Visualization saved: outputs/visualizations/mmwave_simple.png")
        except Exception as e:
            print(f"   ⚠️  Could not create visualization: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

def generate_simulation_data(num_scenarios=5):
    """Generate simulation data for testing"""
    print(f"🧪 Generating {num_scenarios} simulation scenarios...")
    
    try:
        from utils.mmwave_simulator import MMWaveSimulator
        from utils.data_collector import DataCollector  # Your existing class
        
        # Generate mmWave data
        print("   Generating mmWave radar data...")
        simulator = MMWaveSimulator(sampling_rate=100, duration=60)
        scenarios = simulator.generate_scenario_dataset(num_scenarios=num_scenarios)
        
        # Collect weather data using YOUR DataCollector class
        print("   Collecting weather data...")
        collector = DataCollector()
        weather = collector.collect_weather_data("Tokyo")
        
        print(f"   ✅ Generated {len(scenarios)} scenarios")
        print(f"   ✅ Weather data collected")
        print(f"   📁 Data saved to: data/processed/mmwave_vital_signs.json")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Try simplified version
        try:
            print("   Trying simplified simulation...")
            from utils.mmwave_simulator import MMWaveSimulator
            simulator = MMWaveSimulator(sampling_rate=100, duration=30)
            iq_data = simulator.generate_mmwave_iq_data(has_child=True)
            print(f"   ✅ Generated mmWave data: {len(iq_data)} samples")
        except Exception as e2:
            print(f"   ❌ Could not generate simulation data: {e2}")

def run_tests():
    """Run test suite"""
    print("🧪 Running test suite...")
    
    try:
        import pytest
    except ImportError:
        print("   ⚠️  pytest not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
    
    # Run pytest
    result = subprocess.run([
        "pytest", "tests/", 
        "-v"
    ])
    
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    
    return result.returncode

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GuardianSensor - mmWave Radar Child Safety System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py api              # Start API server
  python run.py dashboard        # Start dashboard
  python run.py test             # Run tests
  python run.py simulate         # Generate test data
  python run.py api --port 8080  # Start API on custom port
        """
    )
    
    parser.add_argument(
        "command", 
        nargs="?", 
        default="help",
        choices=["api", "dashboard", "process", "simulate", "test", "help"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host for API server (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--no-reload", 
        action="store_true",
        help="Disable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Execute command
    if args.command == "help":
        parser.print_help()
        
    elif args.command == "api":
        start_api_server(
            host=args.host,
            port=args.port,
            reload=not args.no_reload
        )
        
    elif args.command == "dashboard":
        start_dashboard(port=8501)
        
    elif args.command == "process":
        run_signal_processing()
        
    elif args.command == "simulate":
        generate_simulation_data(num_scenarios=5)
        
    elif args.command == "test":
        run_tests()
        
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Goodbye! GuardianSensor shutting down...")
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)