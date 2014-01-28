#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <omp.h>

unsigned long sequential (std::vector<unsigned long> vect1, unsigned long size) {
    unsigned long totsum = 0; 
    printf("Threads: 1\n"); 
    for (int i = 0; i < size; i++) {
        totsum += vect1[i]; 
    }
    return totsum; 
}

unsigned long parallel (std::vector<unsigned long> vect1, unsigned long size) {
    int nthreads, tid;
    unsigned long totsum = 0; 
    unsigned long p_totsum = 0;
    unsigned long i; 
    #pragma omp parallel private(p_totsum, nthreads,tid, i)
    {
        p_totsum = 0;
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        //if (tid == 0) {
        //    printf("Threads: %d\n",nthreads);
        //}
        i = 0;
        while (i+tid < size) {
            p_totsum += vect1[i+tid];
            i+= nthreads; 
        }
        #pragma omp atomic
        totsum += p_totsum; 
    }
    return totsum; 
}

unsigned long reduce (std::vector<unsigned long> vect1, unsigned long size) {
    unsigned long totsum = 0; 
    #pragma omp parallel for reduction (+:totsum)
        for (int i = 0; i < size; i++) {
            totsum = totsum + vect1[i];
        }
    return totsum; 
}
int main(int argc, char* argv[]) {
    unsigned long size = 0; 
    char type; 
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
    
    gettimeofday(&t1, NULL);
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
    gettimeofday(&t2, NULL);
    printf("totsum: %lu\n", totsum);
    double  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("total time %.4f\n",elapsedTime);
    /*Sequential Benchmark Ends Here*/    
}
