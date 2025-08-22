#!/usr/bin/env python3
"""
GPU and TensorFlow Test Script
This script tests whether TensorFlow can detect and use the GPU for computation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def test_basic_imports():
    """Test basic imports"""
    print("=== Testing Basic Imports ===")
    try:
        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        import scipy
        from scipy import fft
        import sklearn
        print("✓ All basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_tensorflow_gpu():
    """Test TensorFlow GPU availability and functionality"""
    print("\n=== Testing TensorFlow GPU ===")
    
    try:
        import tensorflow as tf
        
        # Print TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs available: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Test GPU computation
            print("\nTesting GPU computation...")
            with tf.device('/GPU:0'):
                # Create random matrices
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                
                # Perform matrix multiplication
                start_time = datetime.now()
                c = tf.matmul(a, b)
                end_time = datetime.now()
                
                print(f"✓ Matrix multiplication on GPU completed")
                print(f"  Time taken: {(end_time - start_time).total_seconds():.4f} seconds")
                print(f"  Result shape: {c.shape}")
            
            return True
        else:
            print("✗ No GPUs detected")
            return False
            
    except Exception as e:
        print(f"✗ TensorFlow GPU test failed: {e}")
        return False

def test_fft_functionality():
    """Test FFT functionality for image processing"""
    print("\n=== Testing FFT Functionality ===")
    
    try:
        # Create a test image with a simple pattern
        x = np.linspace(0, 2*np.pi, 64)
        y = np.linspace(0, 2*np.pi, 64)
        X, Y = np.meshgrid(x, y)
        
        # Create a test pattern (similar to moire lattice)
        test_image = np.sin(3*X) * np.cos(2*Y) + 0.5*np.sin(5*X + Y)
        
        # Perform 2D FFT
        fft_result = np.fft.fft2(test_image)
        fft_magnitude = np.abs(fft_result)
        
        # Find peaks in frequency domain
        from scipy.signal import find_peaks
        flat_magnitude = fft_magnitude.flatten()
        peaks, _ = find_peaks(flat_magnitude, height=np.max(flat_magnitude)*0.1)
        
        print(f"✓ 2D FFT computation successful")
        print(f"  Original image shape: {test_image.shape}")
        print(f"  FFT result shape: {fft_result.shape}")
        print(f"  Number of significant peaks found: {len(peaks)}")
        
        return True
        
    except Exception as e:
        print(f"✗ FFT functionality test failed: {e}")
        return False

def test_cnn_creation():
    """Test CNN creation for the moiré lattice reconstruction task"""
    print("\n=== Testing CNN Creation ===")
    
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        # Create a simple CNN encoder for image-to-lattice parameters
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            # Output: N lattices × 4 parameters (theta, |k|, phi, A)
            layers.Dense(32, activation='linear')  # 8 lattices × 4 params
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Test with dummy data
        dummy_input = tf.random.normal([1, 64, 64, 1])
        output = model(dummy_input)
        
        print(f"✓ CNN model created successfully")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"✗ CNN creation test failed: {e}")
        return False

def test_image_processing():
    """Test image processing capabilities"""
    print("\n=== Testing Image Processing ===")
    
    try:
        import cv2
        from PIL import Image
        
        # Create a synthetic test image
        test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        # Test OpenCV operations
        blurred = cv2.GaussianBlur(test_image, (5, 5), 0)
        edges = cv2.Canny(test_image, 100, 200)
        
        # Test PIL operations
        pil_image = Image.fromarray(test_image)
        resized = pil_image.resize((50, 50))
        
        print(f"✓ Image processing libraries working")
        print(f"  Original shape: {test_image.shape}")
        print(f"  Blurred shape: {blurred.shape}")
        print(f"  Edge detection shape: {edges.shape}")
        print(f"  PIL resize successful: {np.array(resized).shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Image processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("ML Research Environment Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_tensorflow_gpu,
        test_fft_functionality,
        test_cnn_creation,
        test_image_processing
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Basic Imports",
        "TensorFlow GPU",
        "FFT Functionality", 
        "CNN Creation",
        "Image Processing"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ML environment is ready for moiré lattice research.")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
