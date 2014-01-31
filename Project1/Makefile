C=gcc
CFLAGS=-O3 -Wall -std=c99 -Wno-unused-result
CPARFLAGS=-fopenmp
TARGET=parser

OBJS = parser.o \
	main.o

all: $(TARGET)

%.o: %.c %.h
	$(C) $(CFLAGS) -c $<
$(TARGET): $(OBJS)
	$(C) $(CFLAGS) -o $@ $^
clean:
	rm -f *.o parser
	rm -f core*
