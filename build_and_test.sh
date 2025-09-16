#!/bin/bash
# ============================
# cuPDLPx build & test script
# ============================

# 1. request interactive GPU node
#echo ">>> Requesting interactive GPU node..."
#srun --pty --partition=ou_sloan_gpu --cpus-per-task=1 --mem=16G --gres=gpu:1 --time=00:15:00 bash

# 2. load CUDA module
echo ">>> Loading CUDA module..."
module purge
module load cuda/12.4 || { echo "CUDA 12.4 not found, try another version"; exit 1; }

# . show environment info
which nvcc
nvcc --version
echo "CUDA_HOME = $CUDA_HOME"

# 4. build
echo ">>> Cleaning and building..."
make clean
make build

# 5. download test instance if not present
TESTDIR=$HOME/src/cuPDLPx/test
mkdir -p $TESTDIR
if [ ! -f $TESTDIR/2club200v15p5scn.mps.gz ]; then
    echo ">>> Downloading test instance..."
    wget -P $TESTDIR https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz
fi

# 6. run test
echo ">>> Running smoke test..."
./build/cupdlpx $TESTDIR/2club200v15p5scn.mps.gz $TESTDIR

echo ">>> Done! Results saved in $TESTDIR"
