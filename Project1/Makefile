CC=gcc
NVCC=nvcc
#OPT=-O3 -march=corei7 -mtune=corei7-avx  
D= -D BENCHMARK -D PRODUCTION #-D ANNEALER
OPT=-O3
CFLAGS=-Wall -std=gnu99 -lm#-Wno-unused-result -lm  
DEBUG=-g -ggdb -pg
CPARFLAGS=-fopenmp
#RDRAND=-D IVYHASWELL
#ANNEALER=-D NO_ANNEALER
CUARCH = -arch=sm_20
PRODUCTION=
TARGET=placer

TARGETCU=placer_cu
TARGETAN=placer_an
INC=-Iinc/
L=-L /usr/local/cuda/lib64/ -lcudart -lcuda
OBJS = parser.o \
	   solver.o \
	   netlist.o \
	   rand.o \
	   annealer.o \
	   sort.o \
	   main.o 
CUOBJS = jacobi.o \
		 solver_cu.o \
	     solver.o

all: $(TARGET)
cuda: $(TARGETCU)   
annealer: $(TARGETAN) 
%.o: %.c
	$(CC) $(INC) $(D) $(CUDA) $(OPT) $(ANNEALER) $(CFLAGS) $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) -c $<

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(OPT) $(ANNEALER) $(CFLAGS) $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) 

$(TARGETCU): $(OBJS) $(CUOBJS)
	$(CC) -o $@ $(L) $^ $(OPT) $(ANNEALER) $(CFLAGS) $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) 

$(TARGETAN): $(OBJS)
	$(CC) $(INC) $(OPT) $(CFLAGS) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) -c main.c
	$(CC) -o $@ $^ $(OPT) $(ANNEALER) $(CFLAGS) $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) 

solver_cu.o : jacobi.o solver.c
	$(CC) $(INC) -D CUDA $(OPT) $(CFLAGS) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) -c solver.c
	$(NVCC) $(INC) $(CUARCH) -dlink jacobi.o solver.o -o solver_cu.o
%.o: %.cu
	$(NVCC) $(INC) $(CUARCH) -dc  $<
 
clean:
	rm -f *.o parser placer placer_cu
	rm -f qsub\.*
	rm -f core*
