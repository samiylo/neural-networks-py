# Neural Networks Python (TensorFlow)

A collection of Python scripts for working with neural networks using TensorFlow, with support for GPU/Metal acceleration.

## Overview

This repository contains scripts to:
- Verify TensorFlow installation and GPU/Metal support
- Run basic neural network implementations
- Serve as a starting point for TensorFlow-based machine learning projects

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- CUDA drivers (for NVIDIA GPU acceleration, optional)
- Metal (for Apple Silicon acceleration, optional)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/neural-networks-py.git
cd neural-networks-py
```

### 2. Set up a Python virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install tensorflow numpy matplotlib

# For GPU support on NVIDIA systems
pip install tensorflow-gpu

# For Apple Silicon (M1/M2/M3) support
pip install tensorflow-metal
```

## Scripts

### `check-npu` / `py-scripty`

Verifies that TensorFlow is properly installed and can access GPU/Metal acceleration.

```bash
python check-npu
# or
./check-npu
```

### `sanity-check-npu`

Runs a basic neural network implementation to validate the TensorFlow setup.

```bash
python sanity-check-npu
# or
./sanity-check-npu
```

## Usage Examples

### Checking TensorFlow Installation

```python
# Script checks if TensorFlow can access GPU/Metal
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

### Basic Neural Network Example

```python
import tensorflow as tf

# Create a simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train and evaluate (with dummy data in this example)
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)
```

## Troubleshooting

### GPU Not Detected

- Ensure CUDA toolkit and cuDNN are installed (for NVIDIA GPUs)
- Check that your GPU is compatible with TensorFlow
- Verify drivers are up to date

### Memory Issues

If you're experiencing out-of-memory errors:

```python
# Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

## Documentation and Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/api/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [TensorFlow Metal (Apple Silicon) Support](https://developer.apple.com/metal/tensorflow-plugin/)
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
