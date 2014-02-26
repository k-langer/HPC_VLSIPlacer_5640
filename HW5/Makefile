C=gcc
CFLAGS=-Wall -std=gnu99 -Wno-unused-result -lm 
CPARFLAGS=-fopenmp
TARGET=HW5

OBJS = sobel.c

#%.o: %.c
#	$(C)$(CFLAGS) -c $<

all: $(TARGET)

$(TARGET): $(OBJS)
	$(C) $(CFLAGS) -o $@ $^
clean:
	rm -f *.o $(TARGET)
	rm -f core*
test:
	eog result.ppm
