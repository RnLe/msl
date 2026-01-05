# ML Research Environment Setup

This Docker setup provides a complete machine learning research environment with CUDA support for moiré lattice reconstruction research.

## Features

- **CUDA Support**: Latest NVIDIA CUDA base image (12.2) for GPU acceleration
- **TensorFlow**: TensorFlow 2.15 with CUDA support
- **Jupyter Lab**: Full Jupyter Lab environment accessible via web browser
- **Scientific Stack**: NumPy, SciPy, Matplotlib, OpenCV, scikit-learn, and more
- **Signal Processing**: Specialized libraries for FFT analysis and image processing
- **Development Tools**: Black, flake8, pytest, and debugging tools

## Quick Start

### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Container Toolkit** installed on your host system
3. **Docker** and **Docker Compose**

### Installation

1. Navigate to the web directory:
   ```bash
   cd /home/renlephy/msl/web
   ```

2. Build and start the ML research container:
   ```bash
   docker-compose up --build msl_ml_research
   ```

3. Access Jupyter Lab in your browser:
   ```
   http://localhost:8888
   ```

### Directory Structure

The container mounts the following directories:

- `/workspace/notebooks` ← Maps to `../research/multi_moire_construction`
- `/workspace/data` ← Maps to `../data` (create if needed)
- `/workspace/models` ← Persistent volume for trained models
- `/workspace/results` ← Persistent volume for results

### Testing the Environment

1. **Run the environment test script**:
   ```bash
   # Inside the container or from Jupyter
   python test_ml_environment.py
   ```

2. **Open the setup notebook**:
   - Navigate to `moire_lattice_setup.ipynb` in Jupyter Lab
   - Run all cells to verify the environment

### Key Files

- `test_ml_environment.py` - Comprehensive environment testing
- `moire_lattice_setup.ipynb` - Interactive setup and testing notebook
- `docker/ml-research.Dockerfile` - Docker configuration
- `requirements.txt` - Additional Python packages

## Project Overview

### Research Goal
Reconstruct grayscale images as multiple moiré lattices using both:
1. **Algorithmic approach**: FFT-based peak extraction
2. **Learned approach**: CNN encoder + differentiable renderer

### Workflow
1. Input: Grayscale images
2. Output: Array of Bravais lattices with parameters:
   - Base vectors (2D)
   - Hole size
   - Or alternatively: lattice type, rotation angle, hole size

### Implementation Strategy
- **FFT baseline**: 2D FFT → peak picking → lattice parameter mapping
- **Neural network**: CNN encoder → lattice parameters → differentiable renderer
- **End-to-end training**: Reconstruction loss backpropagated through renderer

## Development Commands

### Start Development Environment
```bash
docker-compose up msl_ml_research
```

### Build Only (without starting)
```bash
docker-compose build msl_ml_research
```

### Run in Production Mode
```bash
docker-compose up -d msl_ml_research_prod
```

### Enter Container Shell
```bash
docker-compose exec msl_ml_research bash
```

### View Logs
```bash
docker-compose logs msl_ml_research
```

## GPU Verification

The environment automatically detects and uses available GPUs. To verify:

1. Check GPU availability in the test script
2. Monitor GPU usage: `nvidia-smi`
3. TensorFlow should show GPU devices in the notebook

## Troubleshooting

### GPU Not Detected
- Ensure NVIDIA Container Toolkit is installed
- Check `nvidia-smi` works on host
- Verify Docker can access GPU: `docker run --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi`

### Port Already in Use
- Change port mapping in `docker-compose.yml`:
  ```yaml
  ports:
    - "8889:8888"  # Use different host port
  ```

### Permission Issues
- Ensure your user is in the `docker` group
- Or run with `sudo docker-compose`

## Next Steps

1. **Generate Training Data**: Create synthetic moiré lattice datasets
2. **Implement FFT Baseline**: Classical signal processing approach
3. **Train Neural Model**: End-to-end learning with differentiable renderer
4. **Evaluate Performance**: Compare algorithmic vs. learned approaches
5. **Real Image Testing**: Test on actual grayscale images

## Resources

- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Jupyter Lab Documentation](https://jupyterlab.readthedocs.io/)

---

**Ready to start your moiré lattice reconstruction research!**
