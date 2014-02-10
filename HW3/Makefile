C=mpiCC
#CFLAGS=-O2 -Wall -std=c++11 -fopenmp
CFLAGS=-Wno-deprecated

#%.o: %.c
#	$(C)$(CFLAGS) -c $<

all: Q1 \
	Q2 \
	Q3 \
    Q4

Q1: Q1.cpp 
	$(C) $(CFLAGS) -o $@ $^
Q2: Q2.cpp 
	$(C) $(CFLAGS) -o $@ $^
Q3: Q3.cpp 
	$(C) $(CFLAGS) -o $@ $^
Q4: Q4.cpp 
	$(C) $(CFLAGS) -o $@ $^
clean:
	rm -f *.o HW2
	rm -f core*
	rm -f Q1 Q2 Q3 Q4
	rm -f .host*
	cat logs/output.txt > logs/.output.txt
	rm -f logs/error.txt
	rm -f logs/output.txt 
