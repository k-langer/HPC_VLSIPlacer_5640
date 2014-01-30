C=gcc
CFLAGS=-O0 -Wall
CPARFLAGS=-fopenmp
TARGET=parser

OBJS = parser.o

all: $(TARGET)

%.o: %.c %.h
	$(C) $(CFLAGS) -c $<
$(TARGET): $(OBJS)
	$(C) $(CFLAGS) -o $@ $^
clean:
	rm -f *.o parser
	rm -f core*
