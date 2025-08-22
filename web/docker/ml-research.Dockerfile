# Multi-stage Dockerfile for ML Research Environment
# Base stage with CUDA and Python
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Development stage
FROM base AS development

# Set working directory
WORKDIR /workspace

# Install core ML packages
RUN pip install --break-system-packages --no-cache-dir \
    tensorflow[and-cuda] \
    tensorflow-probability \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    plotly \
    opencv-python \
    scikit-image \
    scikit-learn \
    pandas \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    tqdm \
    pillow \
    imageio \
    h5py

# Install additional packages for signal processing and image analysis
RUN pip install --break-system-packages --no-cache-dir \
    pywavelets \
    spectrum \
    librosa \
    albumentations

# Install development tools
RUN pip install --break-system-packages --no-cache-dir \
    black \
    flake8 \
    pytest \
    ipython \
    ipdb

# Create directories for the project
RUN mkdir -p /workspace/notebooks
RUN mkdir -p /workspace/data
RUN mkdir -p /workspace/models
RUN mkdir -p /workspace/results

# Copy requirements file if it exists (optional)
COPY requirements.txt* /workspace/

# Install additional requirements if the file exists
RUN if [ -f /workspace/requirements.txt ]; then pip install --break-system-packages -r /workspace/requirements.txt; fi

# Set up Jupyter configuration
RUN jupyter lab --generate-config
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "Starting Jupyter Lab..."\n\
echo "Access at: http://localhost:8888"\n\
echo "Working directory: $(pwd)"\n\
echo "Files available:"\n\
ls -la\n\
echo "Starting Jupyter Lab in background..."\n\
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser &\n\
JUPYTER_PID=$!\n\
echo "Jupyter Lab started with PID: $JUPYTER_PID"\n\
echo "Container will stay alive. Use Ctrl+C to stop."\n\
echo "You can also run: docker exec -it <container_name> bash"\n\
\n\
# Keep container alive\n\
tail -f /dev/null\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# Expose Jupyter port
EXPOSE 8888

# Default command - use the startup script
CMD ["/workspace/start.sh"]

# Production stage (minimal for deployment)
FROM base AS production

WORKDIR /app

# Copy only necessary files
COPY requirements.txt* /app/

# Install only production dependencies
RUN pip install --break-system-packages --no-cache-dir \
    tensorflow[and-cuda] \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    scikit-image \
    pandas \
    pillow

# Install additional requirements if the file exists
RUN if [ -f /app/requirements.txt ]; then pip install --break-system-packages -r /app/requirements.txt; fi

# Default command for production
CMD ["python"]
