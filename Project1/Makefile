CC=gcc
CFLAGS=-O0 -Wall -std=gnu99 -Wno-unused-result -g 
CPARFLAGS=-fopenmp
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
	$(CC) $(INC) $(CFLAGS) -c $<
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -f *.o parser placer
	rm -f core*
