"""
EggVision Environment & Hardware Verification Script
Verifies Python version, CUDA GPU acceleration, OpenCV, and AI dependencies.
"""
import sys

def check_environment():
    print("=" * 60)
    print("🥚 EggVision Environment Verification")
    print("=" * 60)
    
    # 1. Python version
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    
    # 2. PyTorch & CUDA check
    try:
        import torch
        print(f"🔥 PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"⚡ CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"💾 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("⚠️ Warning: CUDA is not available. PyTorch will run on CPU.")
    except ImportError:
        print("❌ PyTorch is not installed yet.")

    # 3. OpenCV check
    try:
        import cv2
        print(f"📷 OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV is not installed yet.")

    # 4. Ultralytics check
    try:
        import ultralytics
        print(f"🚀 Ultralytics YOLO Version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics is not installed yet.")

    # 5. Supervision check
    try:
        import supervision as sv
        print(f"🎯 Supervision Version: {sv.__version__}")
    except ImportError:
        print("❌ Supervision is not installed yet.")

    # 6. MLflow check
    try:
        import mlflow
        print(f"📊 MLflow Version: {mlflow.__version__}")
    except ImportError:
        print("❌ MLflow is not installed yet.")

    # 7. DVC check
    try:
        import dvc
        print(f"📦 DVC Version: {dvc.__version__}")
    except ImportError:
        print("❌ DVC is not installed yet.")


    print("=" * 60)

if __name__ == "__main__":
    check_environment()
