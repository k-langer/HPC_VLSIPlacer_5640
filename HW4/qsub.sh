#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
PROGRAM_PATH=/Users/Student/klanger/EECE5640/HW4/
touch output.txt
# Problem 2
${PROGRAM_PATH}vector_cuda 100 >> ${PROGRAM_PATH}output.txt
${PROGRAM_PATH}vector_cuda 200 >> ${PROGRAM_PATH}output.txt
${PROGRAM_PATH}vector_cuda 1000 >> ${PROGRAM_PATH}output.txt
# Problem 3
${PROGRAM_PATH}vector_cuda 128 >> ${PROGRAM_PATH}output.txt
${PROGRAM_PATH}vector_cuda 256 >> ${PROGRAM_PATH}output.txt
${PROGRAM_PATH}vector_cuda 1024 >> ${PROGRAM_PATH}output.txt
