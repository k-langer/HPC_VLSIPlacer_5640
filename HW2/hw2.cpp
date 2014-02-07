#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <omp.h>

unsigned long sequential (const std::vector<unsigned long> & vect1, unsigned long size) {
    unsigned long totsum = 0; 
    /*printf("Threads: 1\n"); */
    //Simply add up all values in the vector
    for (int i = 0; i < size; i++) {
        totsum += (vect1)[i]; 
    }
    return totsum; 
}

unsigned long parallel (const std::vector<unsigned long> & vect1, unsigned long size) {
    int nthreads, tid;
    unsigned long totsum = 0; 
    unsigned long p_totsum = 0;
    unsigned long i; 
    //Start up openmp and provide private varibles to the threads
    #pragma omp parallel private(p_totsum, nthreads,tid, i)
    {
        //Was having trouble with this variable not always starting with the value 0
        p_totsum = 0;
        // Set the rest of the private variables
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        if (tid == 0) {
            printf("Threads: %d\n",nthreads);
        }
        i = 0;
        //The idea here is to split up the adds into nthread adds. 
        // The condition in the while loop prevents any thread from going out of bounds
        while (i+tid < size) {
            p_totsum += (vect1)[i+tid];
            //This indexing prevents any overlapping adds
            i+= nthreads; 
        }
        //The final update requires nthreads attomic writes to totsum
        #pragma omp atomic
        totsum += p_totsum; 
    }
    return totsum; 
}

unsigned long reduce (const std::vector<unsigned long> & vect1, unsigned long size) {
    unsigned long totsum = 0; 
    //Simply using the reduction pragma provided by openmp
    #pragma omp parallel for reduction (+:totsum)
        for (int i = 0; i < size; i++) {
            totsum = totsum + (vect1)[i];
        }
    return totsum; 
}
int main(int argc, char* argv[]) {
    unsigned long size = 0; 
    char type; 
    //arg one is the vector size
    //arg 2 is the kernel to run
    if (argc > 2 && atoi(argv[1])) {
        size =  atol(argv[1]); 
        type = argv[2][0];
    } else {
        printf("Please give the size and parallel type as args\n"
            "Types: p[arallel openmp] s[equnetial] r[eduction with openmp]\n"
            "Exp: ./HW2 1000 p\n");
        return -1;
    }
    timeval t1, t2;
    unsigned long totsum = 0; 
    std::vector<unsigned long> vect1(size,0);
    for (int i = 0; i < size; i++) {
        vect1[i] = i+1;
    }
    //Start timer--I think this will only work on POSIX..     
    gettimeofday(&t1, NULL);
    //Pick kernel
    switch (type) {
        case 'p':
            totsum = parallel (vect1, size);
            break;
        case 's':
            totsum = sequential (vect1, size);
            break; 
        case 'r':
            totsum = reduce (vect1, size);
            break;
        default:
            printf("Types: p[arallel openmp] s[equnetial] r[eduction with openmp]\n");
    }
    //End timer
    gettimeofday(&t2, NULL);
    printf("totsum: %lu\n", totsum);
    //Provide results in milli seconds. Because of bounds won't work if this takes more than 
    // about an hour 
    double  elapsedTime = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
    printf("total time (ms): %.4f\n",elapsedTime);
    /*Sequential Benchmark Ends Here*/    
}
