C=gcc
CFLAGS=-Wall -std=gnu99 -lm  
IGNORE=-Wno-format-contains-nul -Wno-unused-result 
CPARFLAGS=-fopenmp
TARGET=HW5
#DEBUG=-g -ggdb -pg

OBJS = sobel.c

#%.o: %.c
#	$(C)$(CFLAGS) -c $<

all: $(TARGET)

$(TARGET): $(OBJS)
	$(C) $(CFLAGS) $(IGNORE) $(DEBUG) $(CPARFLAGS) -o $@ $^
seq:
	$(C) $(CFLAGS) $(IGNORE) $(DEBUG) $(CPARFLAGS) -o seq org_sobel.c
clean:
	rm -f *.o $(TARGET) result.ppm
	rm -f core*
test:
	rm -f result.ppm && ./$(TARGET) && eog result.ppm
