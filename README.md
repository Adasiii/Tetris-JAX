# Tetris-JAX

This project is a Tetris game implemented using JAX and Equinox, utilizing a Deep Q-Network (DQN) to train an agent to play Tetris. The code includes scripts for training and testing the model, designed for Ubuntu 20.04 with CUDA 12.0 for GPU acceleration.

## Installation

### System Requirements
- **Operating System**: Ubuntu 20.04
- **GPU Driver**: CUDA 12.0
- **Tools**: Miniconda, git

Before starting, ensure that CUDA 12.0 is installed. Refer to the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for instructions on installing CUDA 12.0 on Ubuntu 20.04. After installing CUDA, follow these steps to set up the project environment:

1. **Install Miniconda**  
   If Miniconda is not installed, download and install it from the [Miniconda official website](https://docs.conda.io/en/latest/miniconda.html). Miniconda provides a lightweight Python environment management tool.

2. **Create a Virtual Environment**  
   Use conda to create a Python 3.10 virtual environment:
   ```bash
   conda create -n tetris python=3.10 -y
   ```

3. **Activate the Virtual Environment**  
   Activate the newly created environment:
   ```bash
   conda activate tetris
   ```

4. **Clone the Code Repository**  
   Download the project code from GitHub:
   ```bash
   git clone https://github.com/Adasiii/Tetris-JAX.git
   ```

5. **Install Dependencies**  
   Navigate to the project directory and install the Python dependencies from requirements.txt:
   ```bash
   cd Tetris-JAX
   pip install -r requirements.txt
   ```

**Note**:
- Ensure your GPU driver is compatible with CUDA 12.0. You can check the driver version with `nvidia-smi`.
- If requirements.txt includes `jax[cuda]`, it will attempt to install a JAX version compatible with your system's CUDA. If issues arise, try installing JAX via conda:
  ```bash
  conda install jax jaxlib -c conda-forge
  ```

## Training

To train the DQN model, run the following command:
```bash
python train.py
```

### Training Parameters
`train.py` supports command-line arguments to adjust training settings, such as:
- `--width`: Board width (default: 10)
- `--height`: Board height (default: 20)
- `--batch_size`: Batch size (default: 512)
- `--lr`: Learning rate (default: 1e-3)
- `--num_epochs`: Number of training epochs (default: 3000)

Example: Adjust learning rate and batch size:
```bash
python train.py --lr 0.0005 --batch_size 256
```

During training, the model will be periodically saved to the `trained_models` directory (e.g., `tetris_500.pkl`), and the best-performing model will be recorded.

## Testing

To test the trained model, run the following command:
```bash
python test.py
```

### Testing Parameters
`test.py` supports the following command-line arguments:
- `--width`: Board width (default: 10)
- `--height`: Board height (default: 20)
- `--fps`: Video frame rate (default: 300)
- `--saved_path`: Path to saved models (default: trained_models)
- `--result`: Output video filename (default: result.mp4)

Example: Test a specific model and change the output filename:
```bash
python test.py --saved_path trained_models --result my_tetris_test.mp4
```

Testing will load the specified model (default: `tetris_2000.pkl`) and generate a Tetris gameplay video, saved to the specified path.

## Dependencies
- **JAX and Equinox**: Used for implementing the DQN and neural network.
- **CUDA 12.0**: Provides GPU acceleration support, ensuring JAX can utilize NVIDIA GPUs.
- **requirements.txt**: Lists all Python dependencies, including JAX, numpy, tensorboardX, etc.

If you encounter issues with JAX's CUDA compatibility, try installing specific versions of JAX and jaxlib via conda-forge:
```bash
conda install jax jaxlib -c conda-forge
```

## Notes
- **CUDA Compatibility**: CUDA 12.0 on Ubuntu 20.04 may require manual configuration of NVIDIA drivers. Refer to the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to ensure drivers and toolkit are correctly installed.
- **GPU Detection**: Run the following command to check if JAX detects the GPU:
  ```python
  import jax
  print(jax.devices())
  ```
  The output should include GPU devices (e.g., `[GpuDevice(...)]`).
- **Environment Issues**: If `pip install -r requirements.txt` fails, check version compatibility in requirements.txt or try installing major dependencies via conda.

## FAQs
- **Q: What if the GPU is not being used during training?**  
  A: Ensure CUDA 12.0 and NVIDIA drivers are correctly installed, and check if JAX detects the GPU. If necessary, reinstall JAX:
  ```bash
  pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```
- **Q: What if the model file is not found during testing?**  
  A: Ensure the `trained_models` directory contains `tetris_2000.pkl` or other model files, and verify the `--saved_path` parameter is correct.

## Contributors
This project is developed by [Adasiii](https://github.com/Adasiii). If you have suggestions for improvements or find issues, please submit an issue or pull request.
