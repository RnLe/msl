#!/bin/bash
set -e

# Activate Conda environment
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate msl

echo "Prefix: $CONDA_PREFIX"

# Set compilers
export CC=gcc
export CXX=g++
export F77=gfortran
export MPICC=gcc
export MPICXX=g++

# Flags to prefer Conda environment
export CFLAGS="-I$CONDA_PREFIX/include -fPIC -O3 -march=native"
export CXXFLAGS="-I$CONDA_PREFIX/include -fPIC -O3 -march=native"
export CPPFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"

# Libctl path
export LIBCTL_DIR="$CONDA_PREFIX/share/libctl"

# Create build directory
mkdir -p research/build_src
cd research/build_src

# 1. Build MPB (Serial + Python)
if [ ! -d "mpb" ]; then
    git clone https://github.com/NanoComp/mpb.git
fi
cd mpb
git checkout master
git pull

echo "Building MPB..."
# Force reconf
./autogen.sh --enable-shared --prefix=$CONDA_PREFIX --with-libctl=$LIBCTL_DIR --with-python --without-mpi
make -j4
make install
cd ..

# 2. Build Meep (OpenMP + Python)
if [ ! -d "meep" ]; then
    git clone https://github.com/NanoComp/meep.git
fi
cd meep
git checkout master
git pull

echo "Building Meep..."
# Explicitly point to conda libctl and disable HDF5 since header was missing anyway (or we could fix it)
# We add -fPIC explicitly to flags just in case.
./autogen.sh --enable-shared --prefix=$CONDA_PREFIX --with-libctl=$LIBCTL_DIR --with-python --without-mpi --with-openmp --without-hdf5

make clean
make -j4
make install

echo "Build Complete!"
