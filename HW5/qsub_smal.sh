#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
PROGRAM_PATH=/Users/Student/klanger/EECE5640/HW5/
cd ${PROGRAM_PATH}
# HW5
./HW5_seq
./HW5_par
./HW5_cuda
