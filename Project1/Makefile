CC=gcc
#OPT=-O2 -march=corei7 -mtune=corei7-avx
CFLAGS=-Wall -std=gnu99 -Wno-unused-result -lm 
# DEBUG=-g -ggdb
CPARFLAGS=-fopenmp
#RDRAND=-D IVYHASWELL
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
	$(CC) $(INC) $(OPT) $(CFLAGS) $(RDRAND) $(DEBUG) -c $<

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(OPT) $(CFLAGS) $(RDRAND) $(DEBUG) 
clean:
	rm -f *.o parser placer
	rm -f core*
