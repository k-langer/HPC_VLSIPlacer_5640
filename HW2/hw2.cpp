#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>

int main(int argc, char* argv[]) {
    long size = 0; 
    if (argc > 1 && atoi(argv[1])) {
        size =  atol(argv[1]); 
    } else {
        printf("Please give the size of the vector as a cmd line arg\n");
        return -1;
    }
    timeval t1, t2;
    long totsum = 0; 
    std::vector<long> vect1(size,0);
    for (int i = 0; i < size; i++) {
        vect1[i] = i+1;
    }
    /*Sequential Benchmark Starts here*/
    gettimeofday(&t1, NULL);
    for (int i = 0; i < size; i++) {
        totsum += vect1[i]; 
    }
    gettimeofday(&t2, NULL);
    printf("totsum: %d\n", totsum);
    double  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("total time %.4f\n",elapsedTime);
    /*Sequential Benchmark Ends Here*/
}
