#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
PROGRAM_PATH=/Users/Student/klanger/EECE5640/HW5/
cd ${PROGRAM_PATH}
# HW5
./HW5_seq earth.ppm 1000
./HW5_par earth.ppm 1000
./HW5_cuda earth.ppm 1000
