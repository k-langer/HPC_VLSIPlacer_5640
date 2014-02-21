C=nvcc
CFLAGS=-g -G
TARGET=vector_cuda

OBJS = vector_cuda.cu

#%.o: %.c
#	$(C)$(CFLAGS) -c $<

all: $(TARGET)

$(TARGET): $(OBJS)
	$(C) $(CFLAGS) -o $@ $^
clean:
	rm -f $(TARGET)
	rm -f output.txt
	rm -f qsub.sh.*
test:
	qsub ./qsub.sh

