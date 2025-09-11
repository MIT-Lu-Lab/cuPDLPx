#!/bin/bash
# ============================
# cuPDLPx build & test script
# ============================

# 1. load CUDA module
echo ">>> Loading CUDA module..."
module purge
module load cuda/12.4 || { echo "CUDA 12.3 not found, try another version"; exit 1; }

# 2. show environment info
which nvcc
nvcc --version
echo "CUDA_HOME = $CUDA_HOME"

# 3. build
echo ">>> Cleaning and building..."
make clean
make build

# 4. download test instance if not present
TESTDIR=$HOME/src/cuPDLPx/test
mkdir -p $TESTDIR
if [ ! -f $TESTDIR/2club200v15p5scn.mps.gz ]; then
    echo ">>> Downloading test instance..."
    wget -P $TESTDIR https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz
fi

# 5. run test
echo ">>> Running smoke test..."
./build/cupdlpx $TESTDIR/2club200v15p5scn.mps.gz $TESTDIR

echo ">>> Done! Results saved in $TESTDIR"
