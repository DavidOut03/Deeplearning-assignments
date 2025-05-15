import os
import ctypes

# 🔐 Add CUDA DLL path for TensorFlow to load GPU libraries correctly on Windows
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
os.add_dll_directory(cuda_path)

# ✅ Attempt to load a core CUDA DLL manually (to verify it's accessible)
try:
    ctypes.WinDLL(os.path.join(cuda_path, "cudart64_110.dll"))
    print("✅ cudart64_110.dll loaded manually")
except Exception as e:
    print("❌ Failed to load cudart64_110.dll:", e)

# 🧠 Now safely import TensorFlow
import tensorflow as tf

print("\n🧠 TensorFlow version:", tf.__version__)
print("🔍 Built with CUDA:", tf.test.is_built_with_cuda())
print("⚙️ cuDNN enabled:", tf.test.is_built_with_gpu_support())

# 🧪 Check physical devices
available_devices = tf.config.list_physical_devices()
gpu_devices = tf.config.list_physical_devices('GPU')

print("\n📦 Available devices:", available_devices)
print("📦 GPU devices:", gpu_devices)

# 📊 Optional: Enable memory growth (avoids full GPU allocation)
if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memory growth enabled for GPU")
    except RuntimeError as e:
        print("⚠️ Could not set memory growth:", e)
else:
    print("❌ No GPU detected by TensorFlow")
