#include <stddef.h>
#include <stdlib.h>
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <vector>
main(int argc, char **argv ) {
    MPI_Init(&argc, &argv);
    int i,rank, size, type=99;
    double start_time, end_time, exe_time; 
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    unsigned long v_size = 0;
    std::vector<unsigned long> vect1;
    if (argc > 1 && atoi(argv[1])) {
        v_size =  atol(argv[1]); 
    } else {
        printf("Please give the size and parallel type as args\n"
            "Types: p[arallel openmp] s[equnetial] r[eduction with openmp]\n"
            "Exp: ./HW2 1000 p\n");
        return -1;
    }
    if (rank == 0) { 
        vect1.resize(v_size+(size-v_size%size),0);
        for (int i = 0; i < v_size; i++) {
            vect1[i] = i+1;
        }
    }
    start_time = MPI_Wtime();
    unsigned long bfr[v_size/size]; 
    
    MPI_Scatter(&vect1[0], 1+v_size/size, MPI_UNSIGNED_LONG,
            &bfr, 1+v_size/size, MPI_UNSIGNED_LONG,
            0, MPI_COMM_WORLD);
    unsigned long partial = 0; 
    for (int i = 0; i < v_size/size+1; i++) {
        partial += bfr[i];
    }
    unsigned long sum; 
    MPI_Reduce(&partial, &sum, 1, MPI_UNSIGNED_LONG, 
               MPI_SUM, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if (rank==0) {
        exe_time = end_time - start_time; 
        printf("Sum: %lu\nTime: %f\n",sum,exe_time*1000);
    }
    MPI_Finalize();
}
  
