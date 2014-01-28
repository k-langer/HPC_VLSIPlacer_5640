C=g++
#CFLAGS=-O2 -Wall -std=c++11 -fopenmp
CFLAGS=-fopenmp
TARGET=HW2

OBJS = hw2.cpp

#%.o: %.c
#	$(C)$(CFLAGS) -c $<

all: $(TARGET)

$(TARGET): $(OBJS)
	$(C) $(CFLAGS) -o $@ $^

clean:
	rm -f *.o HW2
	rm -f core*
