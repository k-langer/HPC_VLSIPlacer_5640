CC=gcc
CFLAGS=-O0 -Wall -std=c99 -Wno-unused-result
CPARFLAGS=-fopenmp
TARGET=placer

OBJS = parser.o \
	   netlist.o \
	   solver.o \
   	   main.o

all: $(TARGET)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -f *.o parser placer
	rm -f core*
