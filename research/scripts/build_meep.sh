#!/bin/bash
set -e

# Activate Environment (Ensure this runs in the shell environment if possible, 
# but relying on the caller to have activated or passing vars)
# We assume 'python' is the msl python.
PYTHON=python3

export CONDA_PREFIX=$(dirname $(dirname $(which $PYTHON)))
echo "Using Prefix: $CONDA_PREFIX"

export CC=gcc
export CXX=g++
export FC=gfortran

export CFLAGS="-O3 -march=native -fPIC"
export CXXFLAGS="-O3 -march=native -fPIC"
export FFLAGS="-O3 -march=native -fPIC"

export CPPFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LIBS="-lgfortran -lquadmath"

cd meep

echo "Configuring Meep..."
./configure --prefix=$CONDA_PREFIX \
            --with-python \
            --without-mpi \
            --with-openmp \
            --enable-shared \
            PYTHON=$PYTHON

echo "Building Meep..."
make -j4

echo "Installing Meep..."
make install

echo "Done."
