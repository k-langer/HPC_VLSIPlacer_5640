CC=gcc
CFLAGS=-O0 -Wall -std=c99 -Wno-unused-result
CPARFLAGS=-fopenmp
TARGET=parser

OBJS = parser.o \
	main.o

all: $(TARGET)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -f *.o parser
	rm -f core*
