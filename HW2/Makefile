C=g++
CFLAGS=-g -O2 -Wall -std=c++11
TARGET=HW2

OBJS = hw2.o

%.o: %.c
	$(C) $(CFLAGS) -c $<

all: $(TARGET)

$(TARGET): $(OBJS)
	$(C) $(CFLAGS) -o $@ $^

clean:
	rm -f *.o HW2
	rm -f core*
