CC=gcc
NVCC=nvcc
#OPT=-O3 -march=corei7 -mtune=corei7-avx  
#OPT=-O3
#CFLAGS = -lm
CFLAGS=-Wall -std=gnu99 -Wno-unused-result -lm  
#DEBUG=-g -ggdb -pg
#CPARFLAGS=-fopenmp
#RDRAND=-D IVYHASWELL
#ANNEALER=-D NO_ANNEALER
CUARCH = -arch=sm_20
PRODUCTION=-D PRODUCTION
TARGET=placer
INC=-Iinc/
L=-L /usr/local/cuda/lib64/ -lcudart
OBJS = parser.o \
	   solver.o \
	   netlist.o \
	   rand.o \
	   annealer.o \
	   sort.o \
	   jacobi.o \
	   solver_cu.o \
	   main.o 

all: $(TARGET)
    
%.o: %.c
	$(CC) $(INC) $(OPT) $(ANNEALER) $(CFLAGS) $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) -c $<

$(TARGET): $(OBJS)
	$(CC) -o $@ $(L) $^ $(OPT) $(ANNEALER) $(CFLAGS) $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) 

#$(TARGET): $(OBJS)
#	$(NVCC) -o $@ $^ $(ANNEALER) -lm $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) 

solver_cu.o : jacobi.o solver.o
	$(NVCC) $(INC) $(CUARCH) -dlink jacobi.o solver.o -o solver_cu.o
%.o: %.cu
	$(NVCC) $(INC) $(CUARCH) -dc  $<
 
clean:
	rm -f *.o parser placer
	rm -f qsub\.*
	rm -f core*
