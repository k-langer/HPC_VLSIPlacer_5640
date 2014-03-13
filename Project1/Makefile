CC=gcc
#OPT=-O3 -march=corei7 -mtune=corei7-avx
CFLAGS=-Wall -std=gnu99 -Wno-unused-result -lm  
#DEBUG=-g -ggdb -pg
CPARFLAGS=-fopenmp
#RDRAND=-D IVYHASWELL
#ANNEALER=-D NO_ANNEALER
PRODUCTION=-D PRODUCTION
TARGET=placer
INC=-Iinc/
OBJS = parser.o \
	   solver.o \
	   netlist.o \
	   rand.o \
	   annealer.o \
   	   main.o

all: $(TARGET)
    
%.o: %.c
	$(CC) $(INC) $(OPT) $(ANNEALER) $(CFLAGS) $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) -c $<

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(OPT) $(ANNEALER) $(CFLAGS) $(RDRAND) $(CPARFLAGS) $(PRODUCTION) $(DEBUG) 
clean:
	rm -f *.o parser placer
	rm -f qsub\.*
	rm -f core*
