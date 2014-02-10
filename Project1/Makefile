CC=gcc
CFLAGS=-O3 -Wall -std=gnu99 -Wno-unused-result -lm
DEBUG=-g 
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
	$(CC) $(INC) $(CFLAGS) $(RDRAND) $(DEBUG) -c $<

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(RDRAND) $(DEBUG) 
clean:
	rm -f *.o parser placer
	rm -f core*
