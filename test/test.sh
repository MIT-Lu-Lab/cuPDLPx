#!/bin/bash
# cd ..
cmake -B build
cmake --build build --clean-first

# test
wget -P test/ https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz
./build/cupdlpx test/2club200v15p5scn.mps.gz test/ -v