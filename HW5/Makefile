C=gcc
CPP=g++
NVCC=nvcc
NVFLAGS=-g -G -arch=sm_20 
CFLAGS=-Wall   -std=gnu99
IGNORE=-Wno-format-contains-nul -Wno-unused-result 
CPARFLAGS=-fopenmp
TARGET=HW5
#DEBUG=-g -ggdb -pg

#%.o: %.c
#	$(C)$(CFLAGS) -c $<

all: HW5_seq HW5_par HW5_cuda

HW5_seq: sobel_seq.c
	$(C) $(CFLAGS) $(IGNORE) $(DEBUG) $(CPARFLAGS) -o HW5_seq sobel_seq.c
HW5_par: sobel_par.c
	$(C) $(CFLAGS) $(IGNORE) $(DEBUG) $(CPARFLAGS) -o HW5_par sobel_par.c
HW5_cuda: sobel_cuda.cu
	$(NVCC) $(NVFLAGS) -o HW5_cuda sobel_cuda.cu
	
clean:
	rm -f *.o result.ppm
	rm -f HW5_*
test:
	rm -f result.ppm && ./$(TARGET) && eog result.ppm
